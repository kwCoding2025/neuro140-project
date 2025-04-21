import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import ViTModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import pickle
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import cairosvg

# Import dataset and model from their respective files
from src.utils.data import FloorplanDataset
from src.models import CompositeModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Update paths to be relative to project root and network storage
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "floorplancad-processed" / "pkl"  # Direct to PKL directory
NETWORK_DIR = Path("/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan")
CHECKPOINTS_DIR = NETWORK_DIR / "checkpoints"
RUNS_DIR = NETWORK_DIR / "runs"
SPLIT_DIR = NETWORK_DIR / "split"
EVAL_DIR = NETWORK_DIR / "evaluation_results"
LOGS_DIR = NETWORK_DIR / "logs"

# Create all necessary network directories
for dir_path in [CHECKPOINTS_DIR, RUNS_DIR, SPLIT_DIR, EVAL_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Add at the top of the file with other imports
LOSS_WEIGHTS = (0.5, 0.4, 0.1)  # Define weights as a global constant

def compute_loss(pred, target, weights=(0.5, 0.4, 0.1)):
    """Composite loss function"""
    room_loss = nn.MSELoss()(pred['room_count'], target['room_count'])
    wall_loss = nn.MSELoss()(pred['wall_count'], target['wall_count'])
    coord_loss = nn.L1Loss()(pred['wall_coords'], target['wall_coords'])
    
    return weights[0] * room_loss + weights[1] * wall_loss + weights[2] * coord_loss

def train_epoch(model, loader, optimizer, device, writer, epoch, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_room_loss = 0
    total_wall_loss = 0
    total_coord_loss = 0
    
    # Determine rank for tqdm disabling
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # Default to 0 if not set
    disable_tqdm = local_rank != 0

    # Wrap loader with tqdm, disable based on rank
    progress_bar = tqdm(enumerate(loader), desc='Training', total=len(loader), disable=disable_tqdm)

    for batch_idx, batch in progress_bar:
        images = batch['image'].to(device)
        targets = {k: v.to(device) for k, v in batch['features'].items()}
        
        if (batch_idx % accumulation_steps != 0):
            optimizer.zero_grad()  # Zero gradients at start of accumulation

        predictions = model(images)
        
        # Use compute_loss function for consistency
        loss = compute_loss(predictions, targets, LOSS_WEIGHTS)
        
        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        
        loss.backward()
        
        # Only step after accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Store unnormalized loss for metrics
        total_loss += loss.item() * accumulation_steps
        
        # Calculate individual losses for logging
        room_loss = nn.MSELoss()(predictions['room_count'], targets['room_count'])
        wall_loss = nn.MSELoss()(predictions['wall_count'], targets['wall_count'])
        coord_loss = nn.L1Loss()(predictions['wall_coords'], targets['wall_coords'])
        
        total_room_loss += room_loss.item()
        total_wall_loss += wall_loss.item()
        total_coord_loss += coord_loss.item()
        
        # Log batch metrics only if writer exists (rank 0)
        if writer is not None:
            writer.add_scalar('Batch/room_loss', room_loss.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar('Batch/wall_loss', wall_loss.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar('Batch/coord_loss', coord_loss.item(), epoch * len(loader) + batch_idx)

        # Update tqdm description if running on rank 0
        if local_rank == 0:
             progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

    # Handle any remaining gradients at the end of the epoch
    if len(loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    # Calculate averages
    num_batches = len(loader)
    # Ensure num_batches is not zero to avoid division by zero
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_room_loss = total_room_loss / num_batches
        avg_wall_loss = total_wall_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
    else:
        avg_loss = 0
        avg_room_loss = 0
        avg_wall_loss = 0
        avg_coord_loss = 0

    return {
        'total': avg_loss,
        'room': avg_room_loss,
        'wall': avg_wall_loss,
        'coord': avg_coord_loss
    }

def validate_pkl_file(pkl_file):
    """Validate PKL file contains required data"""
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Validate data structure based on convert_svg.py output
        if not all(key in data for key in ['width', 'height', 'layers']):
            logger.warning(f"Missing required keys in {pkl_file}")
            return False
            
        # Validate layers data
        if not isinstance(data['layers'], dict):
            logger.warning(f"Invalid layers data in {pkl_file}")
            return False
            
        return True
            
    except Exception as e:
        logger.warning(f"Failed to validate {pkl_file}: {str(e)}")
        return False

def save_split_datasets(train_files, val_files, test_files):
    """Save dataset splits to PKL files"""
    split_dir = SPLIT_DIR
    split_dir.mkdir(exist_ok=True)
    
    # Save splits
    with open(split_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_files, f)
    with open(split_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val_files, f)
    with open(split_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test_files, f)
    
    logger.info(f"Saved dataset splits to {split_dir}")

def find_latest_checkpoint(model_name):
    """Find the most recent epoch checkpoint for a model"""
    checkpoints = list(CHECKPOINTS_DIR.glob(f'{model_name}_epoch_*.pt'))
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the max
    epochs = [int(cp.stem.split('_')[-1]) for cp in checkpoints]
    if not epochs:
        return None
        
    latest_epoch = max(epochs)
    return CHECKPOINTS_DIR / f'{model_name}_epoch_{latest_epoch}.pt'

def main():
    try:
        # Set up distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        if local_rank != -1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {device}")

        # Create directories (only on main process)
        if local_rank == 0:
            for dir_path in [CHECKPOINTS_DIR, RUNS_DIR, SPLIT_DIR, EVAL_DIR, LOGS_DIR]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Find all pickle files
        pkl_files = list((DATA_DIR).rglob("*.pkl"))
        
        # Validate files
        valid_files = [f for f in pkl_files if validate_pkl_file(f)]
        logger.info(f"Found {len(valid_files)} valid files out of {len(pkl_files)} total files")
        
        if not valid_files:
            logger.error("No valid files found. Exiting.")
            return
        
        # Split dataset
        train_files, temp_files = train_test_split(valid_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        # Save splits (only on main process)
        if local_rank == 0:
            save_split_datasets(train_files, val_files, test_files)
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.ToTensor(),
        ])
        
        # Only resize and convert to tensor for validation/test
        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        # Create datasets and dataloaders
        train_dataset = FloorplanDataset(train_files, transform=train_transform)
        val_dataset = FloorplanDataset(val_files, transform=eval_transform)
        test_dataset = FloorplanDataset(test_files, transform=eval_transform)
        
        # Set up data loaders with DistributedSampler for multi-GPU
        train_sampler = None
        if local_rank != -1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=local_rank
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)
        
        # Initialize models, optimizer, scheduler
        models_to_train = {
            'resnet50': CompositeModel('resnet50'),
            'vit': CompositeModel('vit')
        }
        
        for model_name, model in models_to_train.items():
            # Check for best checkpoint (to skip training entirely)
            best_checkpoint_path = CHECKPOINTS_DIR / f'{model_name}_best.pt'
            if best_checkpoint_path.exists():
                if local_rank == 0:
                    logger.info(f"Checkpoint {best_checkpoint_path} found for {model_name}. Skipping training.")
                continue
            
            # Check for latest epoch checkpoint (for resuming)
            latest_checkpoint = find_latest_checkpoint(model_name)
            start_epoch = 0
            best_val_loss = float('inf')
            
            # Log model training start only on rank 0
            if local_rank == 0:
                if latest_checkpoint:
                    logger.info(f"Found checkpoint {latest_checkpoint}. Resuming training for {model_name}...")
                else:
                    logger.info(f"Training {model_name} from scratch...")
            
            model = model.to(device)
            
            if local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True
                )
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            
            # Resume from checkpoint if exists
            if latest_checkpoint:
                # Load checkpoint
                map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
                checkpoint = torch.load(latest_checkpoint, map_location=map_location)
                
                # Load model state
                # If using DDP, need to load to the underlying model
                if hasattr(model, 'module'):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer and scheduler state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
                
                # Check if best checkpoint exists to get best val loss
                if best_checkpoint_path.exists():
                    best_checkpoint = torch.load(best_checkpoint_path, map_location=map_location)
                    best_val_loss = best_checkpoint['val_loss']
                
                # Reset scheduler to match the epoch
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=50, last_epoch=start_epoch-1
                )
                
                if local_rank == 0:
                    logger.info(f"Resumed from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
            
            # Only create tensorboard writer on main process
            writer = None
            if local_rank == 0:
                writer = SummaryWriter(RUNS_DIR / model_name)
            
            # Train for remaining epochs (from start_epoch to 50)
            for epoch in range(start_epoch, 50):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                
                train_metrics = train_epoch(model, train_loader, optimizer, device, writer, epoch, accumulation_steps=1)
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(device)
                        targets = {k: v.to(device) for k, v in batch['features'].items()}
                        predictions = model(images)
                        val_loss += compute_loss(predictions, targets, LOSS_WEIGHTS).item()
                
                # Average the validation loss
                val_loss = val_loss / len(val_loader)
                
                # Synchronize val_loss across processes
                if local_rank != -1:
                    val_loss_tensor = torch.tensor([val_loss], device=device)
                    torch.distributed.all_reduce(val_loss_tensor)
                    val_loss = val_loss_tensor.item() / torch.distributed.get_world_size()
                
                # Log metrics - only if writer exists
                if writer is not None:
                    writer.add_scalar('Loss/train', train_metrics['total'], epoch)
                    writer.add_scalar('Loss/val', val_loss, epoch)
                
                # Update save to include start_epoch
                if epoch % 5 == 0 and local_rank == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, CHECKPOINTS_DIR / f'{model_name}_epoch_{epoch}.pt')
                
                scheduler.step()
                
                # Only main process needs to log to console
                if local_rank == 0:
                    logger.info(f'Epoch {epoch}: Train Loss = {train_metrics["total"]:.4f}, Val Loss = {val_loss:.4f}')

    finally:
        if local_rank != -1:
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main() 
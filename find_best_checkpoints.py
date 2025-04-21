# Find and create best checkpoints from existing epoch checkpoints
import torch
import logging
from pathlib import Path
import shutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
NETWORK_DIR = Path("/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan")
CHECKPOINTS_DIR = NETWORK_DIR / "checkpoints"

def find_best_checkpoint(model_name):
    """Find the checkpoint with the lowest validation loss for a model"""
    checkpoints = list(CHECKPOINTS_DIR.glob(f'{model_name}_epoch_*.pt'))
    if not checkpoints:
        logger.error(f"No epoch checkpoints found for {model_name}")
        return None
    
    logger.info(f"Found {len(checkpoints)} checkpoints for {model_name}")
    
    # Track best checkpoint
    best_checkpoint = None
    best_val_loss = float('inf')
    
    # Check validation loss of each checkpoint
    for checkpoint_path in checkpoints:
        try:
            # Load checkpoint to CPU to avoid GPU memory issues
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Get validation loss from checkpoint
            if 'val_loss' in checkpoint:
                val_loss = checkpoint['val_loss']
                epoch = checkpoint['epoch']
                
                logger.info(f"Checkpoint {checkpoint_path.name}: Epoch {epoch}, Val Loss: {val_loss:.4f}")
                
                # Update best if this is better
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = checkpoint_path
            else:
                logger.warning(f"Checkpoint {checkpoint_path} does not contain validation loss")
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
    
    if best_checkpoint:
        logger.info(f"Best checkpoint for {model_name}: {best_checkpoint.name} with val_loss: {best_val_loss:.4f}")
    else:
        logger.error(f"Could not determine best checkpoint for {model_name}")
    
    return best_checkpoint

def main():
    """Create best checkpoint files for all models"""
    # Define model names
    models = ["resnet50", "vit"]
    
    for model_name in models:
        # Check if best checkpoint already exists
        best_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
        if best_path.exists():
            logger.info(f"Best checkpoint for {model_name} already exists at {best_path}")
            continue
        
        # Find best checkpoint
        best_checkpoint = find_best_checkpoint(model_name)
        if best_checkpoint:
            # Create best checkpoint file (either by copy or by saving with modified filename)
            try:
                # Option 1: Copy the file (faster but uses more disk space)
                shutil.copy2(best_checkpoint, best_path)
                
                # Option 2: Load and save with new name (slower but no extra disk space)
                # checkpoint_data = torch.load(best_checkpoint, map_location='cpu')
                # torch.save(checkpoint_data, best_path)
                
                logger.info(f"Created best checkpoint for {model_name} at {best_path}")
            except Exception as e:
                logger.error(f"Error creating best checkpoint for {model_name}: {str(e)}")
        else:
            logger.error(f"No valid checkpoints found for {model_name}")

if __name__ == "__main__":
    main() 
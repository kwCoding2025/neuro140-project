import torch
import json
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights
from transformers import ViTModel
import torch.nn as nn

# Import dataset and model definitions
from src.utils.data import FloorplanDataset # Corrected import path
from src.models import CompositeModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Update paths
PROJECT_ROOT = Path(__file__).parent.parent
NETWORK_DIR = Path("/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan")
EVAL_DIR = NETWORK_DIR / "evaluation_results"
CHECKPOINTS_DIR = NETWORK_DIR / "checkpoints"
SPLIT_DIR = NETWORK_DIR / "split"

# Create directories if they don't exist
for dir_path in [EVAL_DIR, CHECKPOINTS_DIR, SPLIT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from state dict keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict

def convert_to_svg_path(coords):
    """Convert coordinates to SVG path string"""
    if len(coords) < 2:
        return ""
    
    # Start path with first point
    path = f"M {coords[0][0]:.3f},{coords[0][1]:.3f}"
    
    # Add line segments to remaining points
    for x, y in coords[1:]:
        path += f" L {x:.3f},{y:.3f}"
    
    return path

def predictions_to_json(predictions, original_data):
    """Convert model predictions to JSON format matching original schema"""
    output = {
        'width': 100.0,
        'height': 100.0,
        'layers': {}
    }
    
    # Convert wall coordinates back to points
    wall_coords = predictions['wall_coords'].reshape(-1, 2)
    
    # Filter out zero-padding
    valid_coords = wall_coords[~np.all(wall_coords == 0, axis=1)]
    
    # Group coordinates into walls (assuming each wall has 2 points)
    walls = []
    for i in range(0, len(valid_coords)-1, 2):
        start = valid_coords[i]
        end = valid_coords[i+1]
        
        # Skip if points are too close (< 0.01 units)
        if np.linalg.norm(end - start) < 0.01:
            continue
            
        walls.append({
            'd': convert_to_svg_path([start, end]),
            'stroke': "rgb(0,178,0)",  # Wall color
            'stroke-width': "0.1",
            'fill': "none",
            'semantic-id': "17",  # Wall semantic-id
            'instance-id': str(len(walls)),
            'points': [start.tolist(), end.tolist()]
        })
    
    # Add walls to layers
    if walls:
        output['layers']['predicted_walls'] = walls
    
    return output

def evaluate_model(model_path, test_loader, device):
    """Evaluate model performance and generate predictions"""
    model_name = model_path.stem.split('_')[0]  # Extract model name (resnet50 or vit)
    
    # Initialize model with correct backbone
    model = CompositeModel(backbone=model_name)
    model.to(device)
    
    # Load checkpoint with special handling for module prefix
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    
    # Remove module prefix if present (handles DDP-trained models)
    model_state_dict = remove_module_prefix(model_state_dict)
    
    try:
        # Load state dict with strict=False to handle any minor discrepancies
        model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Loaded model from {model_path} with missing keys handled")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    model.eval()
    
    # Evaluate metrics
    room_errors = []
    wall_errors = []
    coord_errors = []
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch['features'].items()}
            
            outputs = model(images)
            
            # Calculate errors
            room_error = torch.abs(outputs['room_count'] - targets['room_count'])
            wall_error = torch.abs(outputs['wall_count'] - targets['wall_count'])
            coord_error = torch.abs(outputs['wall_coords'] - targets['wall_coords'])
            
            room_errors.extend(room_error.cpu().numpy())
            wall_errors.extend(wall_error.cpu().numpy())
            coord_errors.extend(coord_error.mean(dim=1).cpu().numpy())
            
            # Store predictions
            for i in range(len(images)):
                pred = {
                    'room_count': float(outputs['room_count'][i].cpu().numpy()),
                    'wall_count': float(outputs['wall_count'][i].cpu().numpy()),
                    'wall_coords': outputs['wall_coords'][i].cpu().numpy().tolist()
                }
                predictions.append(pred)
    
    # Calculate metrics
    mean_room_error = np.mean(room_errors)
    mean_wall_error = np.mean(wall_errors)
    mean_coord_error = np.mean(coord_errors)
    
    # Normalize errors to [0, 1] range for composite score
    norm_room_error = min(mean_room_error / 5.0, 1.0)  # Assume max error of 5 rooms
    norm_wall_error = min(mean_wall_error / 10.0, 1.0)  # Assume max error of 10 walls
    norm_coord_error = min(mean_coord_error / 0.5, 1.0)  # Normalized to 0.5 range
    
    # Composite score (lower is better)
    composite_score = (0.4 * norm_room_error + 0.4 * norm_wall_error + 0.2 * norm_coord_error)
    
    metrics = {
        'room_count_error': float(mean_room_error),
        'wall_count_error': float(mean_wall_error),
        'coord_error': float(mean_coord_error),
        'composite_score': float(composite_score)
    }
    
    logger.info(f"Metrics for {model_name}:")
    logger.info(f"Room count error: {mean_room_error:.4f}")
    logger.info(f"Wall count error: {mean_wall_error:.4f}")
    logger.info(f"Coordinate error: {mean_coord_error:.4f}")
    logger.info(f"Composite score: {composite_score:.4f}")
    
    return metrics, predictions

def plot_results(metrics, output_dir):
    """Generate visualization plots"""
    # Create bar plot of metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Evaluation Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics.png')
    plt.close()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load test dataset
        with open(SPLIT_DIR / 'test.pkl', 'rb') as f:
            test_files = pickle.load(f)
    except FileNotFoundError:
        logger.error("Test split file not found. Please run training first.")
        return
    
    # Create test dataset and loader
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    test_dataset = FloorplanDataset(test_files, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    
    # Evaluate both models
    models = {
        'resnet50': CHECKPOINTS_DIR / 'resnet50_best.pt',
        'vit': CHECKPOINTS_DIR / 'vit_best.pt'
    }
    
    all_metrics = {}
    
    # Check if model files exist
    for model_name, model_path in models.items():
        if not model_path.exists():
            logger.error(f"Model checkpoint not found: {model_path}")
            continue
            
        logger.info(f"Evaluating {model_name}...")
        metrics, predictions = evaluate_model(model_path, test_loader, device)
        all_metrics[model_name] = metrics
        
        # Save predictions as JSON
        predictions_dir = EVAL_DIR / f'{model_name}_predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        for i, pred in enumerate(predictions):
            with open(predictions_dir / f'prediction_{i}.json', 'w') as f:
                json.dump(pred, f, indent=2)
        
        # Save metrics
        with open(EVAL_DIR / f'{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    if len(all_metrics) >= 2:
        # Calculate improvement only if both models were evaluated
        vit_score = all_metrics['vit']['composite_score']
        resnet_score = all_metrics['resnet50']['composite_score']
        improvement = (resnet_score - vit_score) / resnet_score * 100
        
        report = {
            'resnet50': all_metrics['resnet50'],
            'vit': all_metrics['vit'],
            'vit_improvement': f'{improvement:.2f}%',
            'success_criteria_met': {
                'vit_improvement': improvement > 5,
                'room_accuracy': (1 - all_metrics['vit']['room_count_error']) > 0.7,
                'wall_accuracy': (1 - all_metrics['vit']['wall_count_error']) > 0.6
            }
        }
        
        # Save final report
        with open(EVAL_DIR / 'final_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"ViT improvement over ResNet50: {improvement:.2f}%")
    else:
        logger.warning("Could not compare models - not all checkpoints were found")

if __name__ == '__main__':
    main() 
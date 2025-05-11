# analyze results
import torch
import logging
import json
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# add src to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent # Adjust if script is placed elsewhere
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.data import FloorplanDataset # Assuming FloorplanDataset is in src/utils/data.py

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define base paths
NETWORK_DIR = Path("/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan")
CHECKPOINTS_DIR = NETWORK_DIR / "checkpoints"
EVAL_DIR = NETWORK_DIR / "evaluation_results"
SPLIT_DIR = NETWORK_DIR / "split"
DEFAULT_OUTPUT_DIR = NETWORK_DIR / "analysis_results"

def load_validation_losses(model_name):
    """Load validation losses from all epoch checkpoints for a model."""
    losses = []
    checkpoint_files = sorted(CHECKPOINTS_DIR.glob(f'{model_name}_epoch_*.pt'),
                              key=lambda p: int(p.stem.split('_')[-1]))

    if not checkpoint_files:
        logger.warning(f"No epoch checkpoints found for {model_name}")
        return []

    logger.info(f"Loading validation losses for {model_name} from {len(checkpoint_files)} checkpoints...")
    for ckpt_path in tqdm(checkpoint_files, desc=f"Loading {model_name} losses"):
        try:
            # load to cpu
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'val_loss' in checkpoint and 'epoch' in checkpoint:
                losses.append({
                    'epoch': checkpoint['epoch'],
                    'val_loss': checkpoint['val_loss']
                })
            else:
                logger.warning(f"Checkpoint {ckpt_path.name} missing 'val_loss' or 'epoch' key.")
        except Exception as e:
            logger.error(f"Failed to load or read {ckpt_path.name}: {e}")

    losses.sort(key=lambda x: x['epoch'])
    return losses

def plot_losses(resnet_losses, vit_losses, output_dir):
    """Plot validation losses for both models."""
    plt.figure(figsize=(12, 6))

    if resnet_losses:
        epochs = [l['epoch'] for l in resnet_losses]
        losses = [l['val_loss'] for l in resnet_losses]
        plt.plot(epochs, losses, label='ResNet50 Val Loss', marker='o')

    if vit_losses:
        epochs = [l['epoch'] for l in vit_losses]
        losses = [l['val_loss'] for l in vit_losses]
        plt.plot(epochs, losses, label='ViT Val Loss', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss During Training')
    plt.legend()
    plt.grid(True)
    plot_path = output_dir / 'validation_loss_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved validation loss plot to {plot_path}")

def load_predictions(model_name):
    """Load the single predictions JSON file for a model."""
    # prediction file path
    pred_file = EVAL_DIR / f'{model_name}_predictions.json'

    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return None

    logger.info(f"Loading predictions for {model_name} from {pred_file}...")
    try:
        with open(pred_file, 'r') as f:
            # list of predictions
            predictions = json.load(f)
        logger.info(f"Successfully loaded {len(predictions)} predictions.")
        return predictions
    except Exception as e:
        logger.error(f"Failed to load prediction file {pred_file.name}: {e}")
        return None

def load_ground_truth(test_split_path):
    """Load ground truth features from the test set."""
    try:
        with open(test_split_path, 'rb') as f:
            test_files = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Test split file not found: {test_split_path}")
        return None

    logger.info(f"Loading ground truth features for {len(test_files)} test samples...")
    ground_truths = []
    # temp dataset instance
    temp_dataset = FloorplanDataset(test_files, transform=None)

    for i in tqdm(range(len(test_files)), desc="Extracting ground truth"):
        try:
            # load original data
            with open(test_files[i], 'rb') as f_pkl:
                 data = pickle.load(f_pkl)
            # extract features
            features = temp_dataset._extract_features(data)
            # convert tensors
            gt = {
                'room_count': features['room_count'].item(),
                'wall_count': features['wall_count'].item(),
                'wall_coords': features['wall_coords'].numpy()
            }
            ground_truths.append(gt)
        except Exception as e:
            logger.error(f"Failed to process ground truth for {test_files[i]}: {e}")

    return ground_truths

def analyze_predictions(predictions, ground_truths, model_name):
    """Calculate detailed metrics from predictions and ground truths."""
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        logger.error(f"Prediction/Ground Truth mismatch for {model_name}. Cannot analyze.")
        return None

    logger.info(f"Analyzing {len(predictions)} predictions for {model_name}...")

    room_abs_errors = []
    wall_abs_errors = []
    coord_abs_errors = []
    room_exact_matches = 0
    wall_exact_matches = 0
    room_within_1 = 0
    wall_within_1 = 0

    for pred, gt in zip(predictions, ground_truths):
        # room count
        room_pred = round(pred['room_count']) # round prediction
        room_gt = round(gt['room_count'])
        room_err = abs(room_pred - room_gt)
        room_abs_errors.append(room_err)
        if room_err == 0:
            room_exact_matches += 1
        if room_err <= 1:
            room_within_1 += 1

        # wall count
        wall_pred = round(pred['wall_count']) # round prediction
        wall_gt = round(gt['wall_count'])
        wall_err = abs(wall_pred - wall_gt)
        wall_abs_errors.append(wall_err)
        if wall_err == 0:
            wall_exact_matches += 1
        if wall_err <= 1:
            wall_within_1 += 1

        # wall coordinates
        # ensure numpy array
        coord_pred = np.array(pred['wall_coords'], dtype=np.float32)
        coord_gt = gt['wall_coords']
        if coord_pred.shape == coord_gt.shape:
             coord_err = np.mean(np.abs(coord_pred - coord_gt))
             coord_abs_errors.append(coord_err)
        else:
             # handle shape mismatch
             logger.warning(f"Coordinate shape mismatch: Pred {coord_pred.shape}, GT {coord_gt.shape}. Skipping coord error calculation for this sample.")

    n_samples = len(predictions)
    analysis = {
        'model': model_name,
        'mae_room_count': float(np.mean(room_abs_errors)) if room_abs_errors else 0.0,
        'mae_wall_count': float(np.mean(wall_abs_errors)) if wall_abs_errors else 0.0,
        'accuracy_room_exact': float(room_exact_matches / n_samples),
        'accuracy_wall_exact': float(wall_exact_matches / n_samples),
        'accuracy_room_within_1': float(room_within_1 / n_samples),
        'accuracy_wall_within_1': float(wall_within_1 / n_samples),
        'mae_coord': float(np.mean(coord_abs_errors)) if coord_abs_errors else 0.0,
        'raw_room_errors': room_abs_errors,
        'raw_wall_errors': wall_abs_errors,
    }
    logger.info(f"Analysis complete for {model_name}.")
    logger.info(f"  MAE Room Count: {analysis['mae_room_count']:.4f}")
    logger.info(f"  MAE Wall Count: {analysis['mae_wall_count']:.4f}")
    logger.info(f"  MAE Coordinates: {analysis['mae_coord']:.4f}")
    logger.info(f"  Room Exact Acc: {analysis['accuracy_room_exact']:.2%}")
    logger.info(f"  Wall Exact Acc: {analysis['accuracy_wall_exact']:.2%}")

    return analysis

def plot_error_distributions(resnet_analysis, vit_analysis, output_dir):
    """Plot histograms of room and wall count errors."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # room count errors
    bins = np.arange(max(max(resnet_analysis['raw_room_errors']), max(vit_analysis['raw_room_errors'])) + 2) - 0.5
    axs[0].hist(resnet_analysis['raw_room_errors'], bins=bins, alpha=0.7, label='ResNet50 Room Errors')
    axs[0].hist(vit_analysis['raw_room_errors'], bins=bins, alpha=0.7, label='ViT Room Errors')
    axs[0].set_title('Distribution of Absolute Room Count Errors')
    axs[0].set_xlabel('Absolute Error (|Pred - GT|)')
    axs[0].set_ylabel('Number of Samples')
    axs[0].legend()
    axs[0].grid(axis='y')

    # wall count errors
    bins = np.arange(max(max(resnet_analysis['raw_wall_errors']), max(vit_analysis['raw_wall_errors'])) + 2) - 0.5
    axs[1].hist(resnet_analysis['raw_wall_errors'], bins=bins, alpha=0.7, label='ResNet50 Wall Errors')
    axs[1].hist(vit_analysis['raw_wall_errors'], bins=bins, alpha=0.7, label='ViT Wall Errors')
    axs[1].set_title('Distribution of Absolute Wall Count Errors')
    axs[1].set_xlabel('Absolute Error (|Pred - GT|)')
    axs[1].legend()
    axs[1].grid(axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'error_distribution_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved error distribution plot to {plot_path}")


def main(output_dir):
    """Main function to run analysis and generate plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting analysis. Results will be saved to: {output_dir}")

    model_names = ['resnet50', 'vit']
    all_losses = {}
    all_predictions = {}
    all_analysis = {}

    # 1. training losses
    logger.info("--- Analyzing Training Losses ---")
    resnet_losses = load_validation_losses('resnet50')
    vit_losses = load_validation_losses('vit')
    if resnet_losses or vit_losses:
        plot_losses(resnet_losses, vit_losses, output_dir)
    else:
        logger.warning("No validation losses found for either model. Skipping loss plot.")


    # 2. analyze predictions
    logger.info("--- Analyzing Prediction Errors ---")
    test_split_path = SPLIT_DIR / 'test.pkl'
    ground_truths = load_ground_truth(test_split_path)

    if ground_truths:
        resnet_preds = load_predictions('resnet50')
        vit_preds = load_predictions('vit')

        if resnet_preds:
            resnet_analysis = analyze_predictions(resnet_preds, ground_truths, 'resnet50')
            if resnet_analysis:
                 all_analysis['resnet50'] = {k: v for k, v in resnet_analysis.items() if not k.startswith('raw_')} # exclude raw errors

        if vit_preds:
            vit_analysis = analyze_predictions(vit_preds, ground_truths, 'vit')
            if vit_analysis:
                 all_analysis['vit'] = {k: v for k, v in vit_analysis.items() if not k.startswith('raw_')} # exclude raw errors

        # plot error distributions
        if 'resnet50' in all_analysis and 'vit' in all_analysis:
             plot_error_distributions(resnet_analysis, vit_analysis, output_dir)
        else:
             logger.warning("Could not generate error distribution plot as analysis for one or both models failed.")

        # save summary to json
        summary_path = output_dir / 'prediction_analysis_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(all_analysis, f, indent=2)
            logger.info(f"Saved prediction analysis summary to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis summary: {e}")

    else:
        logger.error("Failed to load ground truth data. Skipping prediction analysis.")

    logger.info("--- Analysis Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze FloorplanCAD model training and prediction results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save analysis plots and summary JSON."
    )
    args = parser.parse_args()
    main(args.output_dir) 
# FloorplanCAD Analysis

This project implements a transformer-based analysis system for the FloorplanCAD dataset, comparing ViT and ResNet50 models for floorplan feature extraction.

## Setup

1. Create conda environment:
```bash
conda create -n floorcad python=3.9
conda activate floorcad
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Directory Structure

- `src/`: Source code
  - `models.py`: Model architectures (ViT, ResNet50)
  - `train.py`: Training pipeline
  - `evaluate.py`: Evaluation metrics and visualization
  - `utils/`: Utility functions
    - `data.py`: Dataset and data loading utilities
- `/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan/`:
  - `checkpoints/`: Model checkpoints
  - `runs/`: TensorBoard logs
  - `logs/`: Training logs
  - `split/`: Train/val/test split files
  - `evaluation_results/`: Evaluation outputs
- `floorplancad-processed/`: Dataset directory
  - `pkl/`: Pickle files containing images and JSON data

## Usage

1. Ensure data is in `data/floorplancad-processed/`
2. Run training and evaluation:
```bash
sbatch train.sh
```

3. Monitor training:
```bash
tensorboard --logdir /n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan/runs/
```

4. Check results in `/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan/evaluation_results/final_report.json`

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (12GB+ VRAM recommended)
- See `requirements.txt` for full list

## Metrics

- Room count accuracy target: >70%
- Wall count accuracy target: >60%
- ViT improvement over ResNet50 target: >5%

## License

MIT 
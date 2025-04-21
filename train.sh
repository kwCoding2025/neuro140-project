#!/bin/bash
#SBATCH --job-name=floorplan_train
#SBATCH --output=/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan/logs/train_%j.log
#SBATCH --error=/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan/logs/train_error_%j.log
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=128000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

# Load Conda and activate environment
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate floorcad

# Create necessary network directories
NETWORK_DIR="/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan"
mkdir -p ${NETWORK_DIR}/{checkpoints,runs,logs,split,evaluation_results}

# Install requirements if needed
pip install -r requirements.txt

# Add project root to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Simplified distributed training setup
export CUDA_VISIBLE_DEVICES=0,1

# Run the training script with torchrun
echo "Starting training..."
srun --cpu_bind=cores --accel-bind=gpu \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    src/train.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Starting evaluation..."
    python -m src.evaluate
else
    echo "Training failed. Skipping evaluation."
    exit 1
fi

# Check if both scripts completed
if [ $? -eq 0 ]; then
    echo "All processes completed successfully."
else
    echo "Evaluation failed."
    exit 1
fi

# Generate final report
echo "Generating report..."
python -c "
import json
from pathlib import Path

# Update paths to network directory
eval_dir = Path('/n/netscratch/tambe_lab/Lab/kweerakoon/checkpoints-floorplan/evaluation_results')
resnet_metrics = json.load(open(eval_dir / 'resnet50_metrics.json'))
vit_metrics = json.load(open(eval_dir / 'vit_metrics.json'))

# Calculate improvement
improvement = ((resnet_metrics['composite_score'] - vit_metrics['composite_score']) 
              / resnet_metrics['composite_score'] * 100)

# Generate report
report = {
    'resnet50': resnet_metrics,
    'vit': vit_metrics,
    'vit_improvement': f'{improvement:.2f}%',
    'success_criteria_met': {
        'vit_improvement': improvement > 5,
        'room_accuracy': (1 - vit_metrics['room_count_error']) > 0.7,
        'wall_accuracy': (1 - vit_metrics['wall_count_error']) > 0.6
    }
}

# Save report
json.dump(report, open(eval_dir / 'final_report.json', 'w'), indent=2)
"

# Confirm completion
echo "Training and evaluation completed. Check ${NETWORK_DIR}/evaluation_results/final_report.json for results."

# Check job stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,State,AllocCPUS,Elapsed,MaxRSS,MaxVMSize,ReqMem,ExitCode 
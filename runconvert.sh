#!/bin/bash
#SBATCH --job-name=convert_svg
#SBATCH --output=convert_svg_%j.log
#SBATCH --error=convert_svg_error_%j.log
#SBATCH --partition=sapphire              # Adjust partition as needed
#SBATCH --time=24:00:00              # Time limit (adjust as needed)
#SBATCH --mem=128000                 # Memory requested (in MB)
#SBATCH --gres=gpu:0                 # Request 2 GPUs, if necessary
#SBATCH --cpus-per-task=8            # Number of CPU cores per task

# Load Conda profile script to enable Conda commands
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh

# Activate the floorplan_cad environment
echo "Activating floorplan_cad environment..."
conda activate floorcad

# Confirm environment activation
if [[ "$CONDA_DEFAULT_ENV" == "floorcad" ]]; then
    echo "Environment floorcad activated successfully!"
else
    echo "Failed to activate the floorcad environment."
    exit 1
fi

# Optionally install any missing Python dependencies (uncomment and modify as needed)
# pip install --quiet cairosvg svgpathtools numpy

# Run the Python script
echo "Starting convert_svg.py..."
python convert_svg.py

# Confirm completion
echo "Script execution completed."

# Check job stats (this will append to output log)
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,State,AllocCPUS,Elapsed,MaxRSS,MaxVMSize,ReqMem,ExitCode
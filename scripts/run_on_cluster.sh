#!/bin/bash -l
#SBATCH --job-name=bea
#SBATCH --mem=12GB
#SBATCH --time=32:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=1
#SBATCH --chdir=$HOME/bea
#SBATCH --output=$HOME/bea/logs/%A_%a.log
#SBATCH --array=0-26

export PYTHONPATH="$HOME/bea"
export HF_API_TOKEN="YOUR KEY"
export OPENAI_API_KEY="YOUR KEY"
export TOKENIZERS_PARALLELISM=false

module load miniconda
source activate YOUR_ENV;

base_path=CODE_DIR/configs/experiments/bea/

config_path="${base_path}${SLURM_ARRAY_TASK_ID}.json"
python scripts/run.py --config ${config_path}
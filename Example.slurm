#!/bin/bash
#SBATCH --job-name=RGB4

#SBATCH -p gpu-2080ti
#SBATCH -A stf
#SBATCH -c 8


#SBATCH --partition=gpu-2080ti
#SBATCH --gpus=1  
#SBATCH --account=stf


#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G

#SBATCH --chdir=.

## module load: path to your virtual environment
source activate "/miniconda3/envs/deeplearningCuda"


## running traning script
python tree-mortality-CNN/paper_train_rgb_CNN.py



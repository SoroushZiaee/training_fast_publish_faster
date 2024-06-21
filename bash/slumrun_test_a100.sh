#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=10G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun ./bash/lab.sh

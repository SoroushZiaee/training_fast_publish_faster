#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=result_a100_1_4.out
#SBATCH --error=error_a100_1_4.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100G
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun ./bash/lab.sh

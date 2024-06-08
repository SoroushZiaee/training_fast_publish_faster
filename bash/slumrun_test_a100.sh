#!/bin/bash
#SBATCH --job-name=mobilenet
#SBATCH --output=mobilenet.out
#SBATCH --error=mobilenet.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun ./bash/lab.sh

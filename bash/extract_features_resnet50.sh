#!/bin/bash
#SBATCH --job-name=extract_features_resnet50
#SBATCH --output=extract_features_resnet50.out
#SBATCH --error=extract_features_resnet50.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python extract_layer_features.py --model resnet50 --task lamem
srun python extract_layer_features.py --model resnet50 --task imagenet


#!/bin/bash
#SBATCH --job-name=extract_features_vgg19
#SBATCH --output=extract_features_vgg19.out
#SBATCH --error=extract_features_vgg19.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python extract_layer_features.py --model vgg19 --task imagenet
srun python extract_layer_features.py --model vgg19 --task lamem


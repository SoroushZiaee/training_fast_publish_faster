#!/bin/bash
#SBATCH --job-name=Imagenet_resnet18
#SBATCH --output=Imagenet_resnet18.out
#SBATCH --error=Imagenet_resnet18.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Running ResNet18"
python /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/training_scripts_resnets.py
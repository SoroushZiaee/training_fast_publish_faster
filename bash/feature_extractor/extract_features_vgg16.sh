#!/bin/bash
#SBATCH --job-name=extract_features_vgg16
#SBATCH --output=extract_features_vgg16.out
#SBATCH --error=extract_features_vgg16.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

# srun python -m memory_profiler python extract_layer_features.py --model vgg16 --task imagenet
# srun python -m memory_profiler python extract_layer_features.py --model vgg16 --task lamem
# srun python -m memory_profiler extract_layer_features.py --model vgg16 --task lamem_shuffle
# srun python -m memory_profiler extract_layer_features.py --model vgg16 --task lamem_pretrain_freeze
# srun python -m memory_profiler extract_layer_features.py --model vgg16 --task lamem_pretrain_no_freeze
srun python -m memory_profiler extract_layer_features.py --model vgg16 --task lamem_random_pretrain_no_freeze




#!/bin/bash
#SBATCH --job-name=calculate_cka
#SBATCH --output="calculate_cka-%A_%a.out"
#SBATCH --error=calculate_cka.err
#SBATCH --array=1-66
#SBATCH --cpus-per-task=5
#SBATCH --time=00:10:00
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/calculate_cka.sh
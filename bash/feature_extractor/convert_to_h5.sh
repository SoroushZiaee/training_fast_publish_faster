#!/bin/bash
#SBATCH --job-name=convert_to_h5
#SBATCH --output="convert_to_h5_%a.out"
#SBATCH --error="convert_to_h5_%a.err"
#SBATCH --array=1-9
#SBATCH --cpus-per-task=5
#SBATCH --time=00:10:00
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun python load_feature_pkl_files_and_convert_to_h5.py --dst_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features_h5 --src_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features --node_id ${SLURM_ARRAY_TASK_ID}



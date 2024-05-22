#!/bin/bash

# Set up the environment
module --force purge
module load StdEnv/2020
module load python/3.9.6
module load ipykernel/2022a
module load gcc/9.3.0
module load cuda/11.4
module load opencv/4.5.5
module load scipy-stack/2022a
#!/bin/bash

sbatch ./bash/extract_features_alexnet.sh
sbatch ./bash/extract_features_inception_v3.sh
sbatch ./bash/extract_features_resnet50.sh
sbatch ./bash/extract_features_resnet101.sh
# sbatch ./bash/extract_features_vgg16.sh
sbatch ./bash/extract_features_vgg19.sh
sbatch ./bash/extract_features_efficient_v2_s.sh
sbatch ./bash/extract_features_vit_b_32.sh
sbatch ./bash/extract_features_vit_b_16.sh
sbatch ./bash/extract_features_resnet18.sh
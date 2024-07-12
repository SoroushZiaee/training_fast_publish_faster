#!/bin/bash

# TODO: add the vgg16
# model_names=("resnet50" "resnet101" "resnet18" "vgg19" "alexnet" "inception_v3" "vit_b_16" "vit_b_32" "efficientnet_v2_s")
model_names=("resnet50")

echo "Running with model_names=${model_names[@]} on job_id=${SLURM_ARRAY_TASK_ID}"
for model_name in "${model_names[@]}"; do
  echo "Running with model_name=${model_name}"
  python calculate_cka_for_resnet50.py \
    --model_name=${model_name} \
    --src_path=/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features \
    --dst_path=/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/cka_results \
    --job_id ${SLURM_ARRAY_TASK_ID}
done

import torch as ch


stats_tensor = ch.load("/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/datasets/LaMem/support_files/lamem_mean_std_tensor.pt")
print(f"{stats_tensor = }")
LAMEM_MEAN, LAMEM_STD = stats_tensor[:3], stats_tensor[3:]
print(f"{LAMEM_MEAN = }")
print(f"{LAMEM_STD = }")
#!/usr/bin/env python3

import numpy as np
import h5py
import os

def unbiased_HSIC(K, L):
  '''Computes an unbiased estimator of HISC. This is equation (2) from the paper'''

  #create the unit **vector** filled with ones
  n = K.shape[0]
  ones = np.ones(shape=(n))

  #fill the diagonal entries with zeros 
  np.fill_diagonal(K, val=0) #this is now K_tilde 
  np.fill_diagonal(L, val=0) #this is now L_tilde

  #first part in the square brackets
  trace = np.trace(np.dot(K, L))

  #middle part in the square brackets
  nominator1 = np.dot(np.dot(ones.T, K), ones)
  nominator2 = np.dot(np.dot(ones.T, L), ones)
  denominator = (n-1)*(n-2)
  middle = np.dot(nominator1, nominator2) / denominator
  
  
  #third part in the square brackets
  multiplier1 = 2/(n-2)
  multiplier2 = np.dot(np.dot(ones.T, K), np.dot(L, ones))
  last = multiplier1 * multiplier2

  #complete equation
  unbiased_hsic = 1/(n*(n-3)) * (trace + middle - last)

  return unbiased_hsic

def CKA(X, Y):
  '''Computes the CKA of two matrices. This is equation (1) from the paper'''
  
  nominator = unbiased_HSIC(np.dot(X, X.T), np.dot(Y, Y.T))
  denominator1 = unbiased_HSIC(np.dot(X, X.T), np.dot(X, X.T))
  denominator2 = unbiased_HSIC(np.dot(Y, Y.T), np.dot(Y, Y.T))

  cka = nominator/np.sqrt(denominator1*denominator2)

  return cka

def get_all_file_paths(folder):
    lamem_shuffle_paths = []
    lamem_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            if 'shuffle' in file:
                lamem_shuffle_paths.append(full_path)
                print(f"Shuffle file added: {full_path}")  # Debugging output
            else:
                lamem_paths.append(full_path)
                print(f"Regular file added: {full_path}")  # Debugging output
    return lamem_paths, lamem_shuffle_paths

def load_h5(file_path):
    with h5py.File(file_path, 'r') as file:
        data = file['features'][:]
        mf = np.array(data)
    
    return mf


def get_CKA(model, layer, lamem_shuffle_paths, lamem_paths): 
    # Construct expected file names
    shuffle_file_name = f'{model}_lamem_shuffle_{layer}.h5'
    normal_file_name = f'{model}_lamem_{layer}.h5'

    # Search for files in the paths list
    shuffle_path = [path for path in lamem_shuffle_paths if shuffle_file_name in path.split('/')[-1]]
    normal_path = [path for path in lamem_paths if normal_file_name in path.split('/')[-1]]

    print(f"Searching for {shuffle_file_name} in shuffle paths.")
    print(f"Searching for {normal_file_name} in normal paths.")
    print(f"Found shuffle path: {shuffle_path}")
    print(f"Found normal path: {normal_path}")

    if shuffle_path and normal_path:
        print("Loading files...")
        try:
            lamem_shuffle_layer = load_h5(shuffle_path[0])
        except OSError as e:
            print("Failed to read file:", e)
            print(f"{shuffle_path = }")
            
        
        try:
            lamem_layer = load_h5(normal_path[0])
        except OSError as e:
            print(f"Failed to read file:", e)
            print(f"{normal_path = }")
            
        # cka_value = CKA(lamem_layer, lamem_shuffle_layer)
        cka_value = 0
        return cka_value
    else:
        print("Files not found or incomplete pair.")
        return None


def get_pool_layers(model_name: str):
    pool_layers_dict = { 
        
        "resnet101": ['maxpool', 'layer1.1.add', 'layer2.0.add', 'layer2.3.add', 'layer3.1.add', 'layer3.4.add', 'layer3.7.add', 'layer3.10.add', 'layer3.13.add', 'layer3.16.add'],

        "vgg19":  ['features.4', 'features.9','features.18','features.27','features.36'],
        
        "inception_v3": ['maxpool1', 'maxpool2', 'Mixed_5b.avg_pool2d', 'Mixed_5c.avg_pool2d', 'Mixed_5d.avg_pool2d', 'Mixed_6b.avg_pool2d', 'Mixed_6c.avg_pool2d', 'Mixed_6d.avg_pool2d', 'Mixed_6e.avg_pool2d', 'AuxLogits.avg_pool2d', 'AuxLogits.adaptive_avg_pool2d', 'Mixed_7b.avg_pool2d', 'Mixed_7c.avg_pool2d'],
        
        "alexnet": [
            "x",
            "features.1",
            "features.3",
            "features.6",
            "features.8",
            "features.11",
            "features.12",
            "avgpool",
            "classifier.1",
            "classifier.3",
            "classifier.6",
        ],
        
        "resnet50": ['maxpool', 'layer1.0.add', 'layer1.2.add', 'layer2.0.add', 'layer2.2.add', 'layer3.0.downsample.0', 'layer3.1.add', 'layer3.3.add', 'layer3.5.add', 'layer4.0.add'],
        "resnet18": ['maxpool', 'layer1.0.add', 'layer1.1.add', 'layer2.0.add', 'layer2.1.add', 'layer3.0.add', 'layer3.1.add', 'layer4.0.add', 'layer4.1.add', 'avgpool'],
        
        "efficientnet_v2_s": ['features.1.0.add', 'features.2.2.add', 'features.3.2.add', 'features.4.2.add', 'features.4.5.add', 'features.5.3.add', 'features.5.6.add', 'features.6.1.add', 'features.6.4.add', 'features.6.7.add'],
    
        }
    return pool_layers_dict.get(model_name, [])

folder = '/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/pool_layers_h5'

lamem_paths, lamem_shuffle_paths = get_all_file_paths(folder)

models = ['alexnet']

model_cka_list = []
for model_name in models:
    layer_cka_list = []

    pool_layers = get_pool_layers(model_name)
    for layer in pool_layers:
        cka_value = get_CKA(model_name, layer, lamem_shuffle_paths, lamem_paths)
        layer_cka_list.append(cka_value)
    
    model_cka_list.append(layer_cka_list)

print (model_cka_list)


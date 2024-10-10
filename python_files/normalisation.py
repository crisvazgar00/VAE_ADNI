from load_database import find_pet, load_img_nii
import yaml
import numpy as np
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random
import torch
import nibabel as nib 
import pandas as pd 
#from tqdm import tqdm
#from scipy.ndimage import binary_erosion, binary_dilation
#from skimage import filters


#_______________________________________

#       NORMALIZATION FUNCTIONS
#_______________________________________


def normalization_min(img_list):
    """
    This function performs the normalization of a list of images.
    Normalization is performed using the mean of the 5% of voxels with
    more intensity. We delimit by 1 using minimum between normalized voxel
    and 1.
    """
    img_norm_list = []
    
    for img in img_list:
        img_flat = img.flatten()
        max_voxels = np.nanpercentile(img_flat, 95)
        mean_max_voxels = np.nanmean(img_flat[img_flat>=max_voxels])
        img_norm = np.minimum(1, img/mean_max_voxels)
    
        img_norm_list.append(img_norm)
             
    return img_norm_list



def normalization_cerebellum(config, img_list):
    """
    Function for loading a template from AAL3 (ROI_MNI_V7.nii).
    It extracts the cerebellum region and load it into an array.
    
    Normalization is performed by applying cerebellum template as
    a mask to an image, performing intensity average of cerebellum region
    and dividing volume by average.
    
    This is done because cerebellum is NOT affected by Alzheimer's disease.
    
    Volumes must be corregistered to single_subj_T1.nii MNI space as
    indicated in the original AAL article.
        Args:
            -img_list: List of volumes (corregistered) to normalize
            
        Output:
            -img_list_norm: List of volumes normalized in intensity
            according to cerebellum average 
    
    """
    config_file = 'config.yaml'

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    path = config['loader']['load_temp']
    temp_file = nib.load(path)
    temp = np.zeros((90,110,90))
    #We use 1:, :-1, and 1: because we want shape (90, 110, 90) instead of (91, 109, 91)
    temp[:,:-1,:] = temp_file.get_fdata()[1:,:,1:]
    
    #Set everything to 0 except for cerebellum labels (from 95 to 120)
    temp_cerebellum = np.zeros_like(temp)
    
    temp_cerebellum[np.logical_and(temp >= 95, temp <= 120)] = temp[np.logical_and(temp >= 95, temp <= 120)]
    #Transform to binary to obtain mask
    mask = np.zeros_like(temp_cerebellum)
    mask[temp_cerebellum != 0] = 1.0
    
    img_list_norm = []
    for img in img_list:
        #img_masked is a volume which ONLY contains the cerebellum of a patient
        img_masked = np.multiply(mask, img)
        #Get nonzero values of img once mask is applied (to not include 0s in average)
        nonzero_img_norm = img_masked[img_masked != 0] 
        #Compute average of cerebellum, normalise and add to list. Nanmean to avoid NaN values
        norm = np.nanmean(nonzero_img_norm)
        #print(f'norm of iter {i} is: {norm}')
        img_norm = img / norm
        img_list_norm.append(img_norm)
    
    
    return img_list_norm















"""
config_file = 'config.yaml'
    
    #LOAD HYPERPARAMETERS FROM CONFIG FILE

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

folder = config['loader']['load_folder']['folder']
prefix = config['loader']['load_folder']['prefix']
extension = config['loader']['load_folder']['extension']
target_folder_name = config['loader']['load_folder']['target_folder_name']
    


imgs_paths = find_pet(folder, prefix, extension, target_folder_name)
#imgs_NIFTI = load_NIFTI(imgs_paths)
imgs_list = load_img_nii(imgs_paths)

print(f'Loading terminated')


imgs_norm_cerebelum = normalization_cerebellum(config, imgs_list)
print(f'Cerebelum normalization terminated')

imgs_norm_min = normalization_min(imgs_list)
print(f'Min normalization terminated')

batch_size = 10
rand_idx = random.sample(range(len(imgs_list)), batch_size)

batch_cerebelum = [imgs_norm_cerebelum[i] for i in rand_idx]
batch_cerebelum = np.array(batch_cerebelum)
batch_cerebelum = torch.from_numpy(batch_cerebelum)
batch_cerebelum = batch_cerebelum.unsqueeze(1)
batch_min = [imgs_norm_min[i] for i in rand_idx]
batch_min = np.array(batch_min)
batch_min = torch.from_numpy(batch_min)
batch_min = batch_min.unsqueeze(1)
print(f'Shape of batch_cerebelum: {batch_cerebelum.shape}, shape of batch_min: {batch_min.shape}')

make_grid_recon(batch_cerebelum, 'norm_cerebelum')
make_grid_recon(batch_min, 'norm_min')

"""
        
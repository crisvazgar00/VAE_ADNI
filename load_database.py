import torch
import os
import glob
import nibabel as nib
import random
import matplotlib.pyplot as plt
import ipyvolume as ipv
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
    Script for finding _pet.nii files inside a main_folder. 

    ARGUMENTS:
    
        -Main folder: -Chosen by hand when running script- folder that contains
        all subfolders and files where data is stored.
    
        -Prefix: prefix of the file we want to find. Usually 'sub', 'wsub',
        'rsub', etc.
    
        -Extension: Always set to '.nii' -NIfTI format-

        -targetFolderName: Name of folder where target files are contained.
        This is optional if we set Extension to be '_pet.nii' (for PET) or
        '_T1w.nii' (for MRI).

"""

def find_pet(folder, prefix, extension, target_folder_name):
    file_list = []
    
    #Get current folder name
    current_folder_name = os.path.basename(folder)
    
    #Check if current folder is target folder
    if current_folder_name == target_folder_name:
        
        #Pattern string of the files we look for
        pattern = os.path.join(folder, f"{prefix}*{extension}")
        #Find files with the patern prefix*extension
        files = glob.glob(pattern)
        #Add to list
        file_list.extend(files)
                
    #Get sub-folders inside current folder
    #It gets the path (f.path) of every f that is a folder if f is a directory and its name is not . or ..            
    sub_folders = [f.path for f in os.scandir(folder) if f.is_dir() and f.name not in {'.', '..'}]
    for sub_folder in sub_folders:
        #Call function for each subfolder to check if there are prefix*extension files inside
        sub_folder_files = find_pet(sub_folder, prefix, extension, target_folder_name)
        file_list.extend(sub_folder_files)
    
    return file_list

def load_img_nii(file_list):
    #img_dic = {}
    img_list = []
    for file_path in file_list:
        img = nib.load(file_path)
        img = img.get_fdata()
        img[np.isnan(img)] = 0
        #dim = img.shape
        #print(dim)
        #img_dic[file_path] = (img, dim)
        img_list.append(img)
    return  img_list




def normalization_cerebellum(img_list):
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
    path = "C:/Program Files/spm12/toolbox/AAL3/ROI_MNI_V7.nii"
    temp = nib.load(path)
    temp = temp.get_fdata()
    
    #Set everything to 0 except for cerebellum labels (from 95 to 120)
    temp_cerebellum = np.zeros_like(temp)
    
    temp_cerebellum[np.logical_and(temp >= 95, temp <= 120)] = temp[np.logical_and(temp >= 95, temp <= 120)]
    #Transform to binary to obtain mask
    mask = np.zeros_like(temp_cerebellum)
    mask[temp_cerebellum != 0] = 1
    
    img_list_norm = []
    for img in img_list:
        #img_mask is a volume which ONLY contains the cerebellum of a patient
        img_mask = np.multiply(mask, img)
        img_mask[np.isnan(img_mask)] = 0
        #Get nonzero values of img once mask is applied (to not include 0s in average)
        nonzero_img_norm = img_mask[img_mask != 0] 
        #Compute average of cerebellum, normalise and add to list. Nanmean to avoid NaN values
        norm = np.nanmean(nonzero_img_norm)
        #print(f'norm of iter {i} is: {norm}')
        img_norm = img / norm
        img_list_norm.append(img_norm)
    
    
    return img_list_norm




def split_database(img_list, splits=None):
    """
    This function splits the image dataset into three sets:
    training, evaluation & test.
        Args:
            -img_list: list of preprocessed volumes.
            -splits: rates of each sate (default to 70%, 15% & 15%).
        
        Output:
            -train_list: list of volumes for training.
            -eval_list: list of volumes for evaluation.
            -test_list: list of volumes for test.
    
    """
    
    if splits is None:
        splits = [0.7, 0.15, 0.15] #train, eval, test
        
    if sum(splits) != 1.0:
        raise ValueError("split rates must sum to 1.0")
    
    N = len(img_list)
    shuffled_list = sorted(img_list, key=lambda x: random.random())
    
    n_train = round(splits[0]*N)
    n_eval = round(splits[1]*N)
    #This way we don't lose any image due to approximation of round()
    n_test = N - n_train - n_eval
    
    if n_train + n_eval + n_test != N:
        raise ValueError("Error in splitting dataset: sum of splits does not match dataset size")
    
    train_list = shuffled_list[0 : n_train] 
    eval_list = shuffled_list[n_train : n_train + n_eval]
    test_list = shuffled_list[n_train + n_eval : n_train + n_eval + n_test]
    
    return train_list, eval_list, test_list
    
    



#folder = 'C:/Users/Cristobal/Desktop/DATOS_PRUEBA'
#prefix = 'r'
#extension = '.nii'
#target_folder_name = 'pet'

#file_list = find_pet(folder, prefix, extension, target_folder_name)
#img_list = load_img_nii(file_list)
#img_list_norm = normalization_cerebellum(img_list)

#mean_img_arr = []
#for img in img_list_norm:
#    mean = np.nanmean(img)
#    mean_img_arr.append(mean)
    
#print(mean_img_arr)
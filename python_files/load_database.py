import torch
import os
import glob
import nibabel as nib
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml
import pandas as pd


#_________________________________________________

#FIND PATHS, LOAD FILES AND PREPROCESS IMAGES
#_________________________________________________


def find_pet(folder, prefix, extension, target_folder_name):
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
    img_list = []
    for file_path in file_list:
        img = np.zeros((90, 110, 90))
        img_file = nib.load(file_path)
        img[:,:-1,:] = img_file.get_fdata()[1:, :, 1:]
        img[np.isnan(img)] = 0
        img_list.append(img)
    return img_list



#________________________________________________

#SPLIT DATABASE INTO TRAIN, EVAL AND TEST SETS
#________________________________________________


def split_database(img_list, splits=None):
    """
    This function splits the image dataset into three sets:
    training, evaluation & test.
        Args:
            -img_list: list of preprocessed volumes.
            -splits: rates of each sate (default to 70%, 15% & 15%).
        
        Output:
            -train_list: list of volumes for training.
            -eval_list: list of volumes for evaluation. (unused)
            -test_list: list of volumes for test.
    
    The way of splitting is shuffling the entire dataset and then asign n_train first
    subjects to train_set, then the next n_eval subjects to train_eval and then the
    last n_test subjects to test_set.
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
    
    if n_train + n_eval + n_test != N: #Check that there is no subjects left
        raise ValueError("Error in splitting dataset: sum of splits does not match dataset size")
    
    train_list = shuffled_list[0 : n_train] 
    eval_list = shuffled_list[n_train : n_train + n_eval]
    test_list = shuffled_list[n_train + n_eval : n_train + n_eval + n_test]
    
    return train_list, eval_list, test_list
    
    
    #____________________________________________________________________________
    
    #FUNCTIONS FOR FILTERING ADNIMERGE DATASET AND MERGE WITH AVAILABLE SUBJECTS
    #____________________________________________________________________________
    
    
def transform_string(tuple_id_ses):
    """This function transforms tuple strings of format ('sub-ADNIXXXSXXXX', 'ses-MXXX') into
    format ('XXX_S_XXXX', 'mXX') to match ADNIMERGE dataset. 
    
    The function extracts each element of the tuple into 'id' ad 'ses'. It removes "sub-ADNI" 
    from id and "ses-" from 'ses'. Then, to match ADNIMERGE, 'M000' is transformed into 'bl'.
    In any other case we transform from 3 digit format 'MXXX' into 2 digit format. Finally
    change from format 'XXXSXXXX' to 'XXX_S_XXXX' and merge with the transformed 'mXX'.
    """ 
    transform_id_ses_tuple = []
    for id_ses in tuple_id_ses:
        id = id_ses[0]
        ses = id_ses[1]
        id = id.replace("sub-ADNI", "")
        transform_ses = ses.replace("ses-", "")
        if transform_ses == 'M000': 
            transform_ses = 'bl'
        else:
            transform_ses = f"{int(transform_ses[1:]):02d}"
            transform_ses = 'm'+transform_ses
        transform_id_ses_tuple.append((f"{id[:3]}_{id[3:4]}_{id[4:]}", transform_ses))
    return transform_id_ses_tuple




def extract_id_ses_from_path(imgs_paths):
    """
    This function extracts a string from the path of a file for a list of paths.
    First, it obtains each part of the path divided by '/' and then extract the
    parts corresponding to 'sub-ADNI' and 'ses-M' and add them to a tuple.
    """
    tuple_id_ses = []
    
    for path in imgs_paths:
        parts = path.replace("\\", "/").split("/")
    
        sub_id_part = next((part for part in parts if part.startswith("sub-ADNI")), None)
        ses_part = next((part for part in parts if part.startswith("ses-M")), None)
    
        tuple_id_ses.append((sub_id_part, ses_part))
    return tuple_id_ses



def merge_id_ses_to_ADNIMERGE(transformed_id_ses_list, ADNIMERGE_df, feature):
    """
    Args:
        -transformed_id_ses_list: list of tuples ('XXX_S_XXXX', 'mXX').
        This list represents the available subjects in the database.
        
        -ADNIMERGE_df: full dataset of subjects
        
    This function merges the available dataset subjects and the full dataset to
    obtain a dataframe with information of ONLY available subjects.
    """
    df = ADNIMERGE_df
    #Transform list of tuples into DataFrame
    df_id_ses = pd.DataFrame(transformed_id_ses_list, columns = ['PTID', 'VISCODE'])
    #Merge list of tuples and full dataset
    df_ADNI_BIDS = pd.merge(df, df_id_ses, on = ['PTID', 'VISCODE'])
    #Drop all columns except ID, Session and ADAS13 or Ventricles
    #if feature == 'ADAS13':
    #    df_ADNI_BIDS_id_ses_ADAS = df_ADNI_BIDS[['PTID', 'VISCODE', 'ADAS13']]
    #    return df_ADNI_BIDS_id_ses_ADAS
    #elif feature == 'Ventricles':
    #    df_ADNI_BIDS_id_ses_VENTRICLES = df_ADNI_BIDS[['PTID', 'VISCODE', 'Ventricles']]
    #    return df_ADNI_BIDS_id_ses_VENTRICLES
    df_ADNI_BIDS_id_ses_feature = df_ADNI_BIDS[['PTID', 'VISCODE', f'{feature}']]
    return df_ADNI_BIDS_id_ses_feature


#____________________________________________________________________

#FUNCTIONS FOR MERGING AND SPLITTING LISTS TO MATCH ID WITH PATIENT
#____________________________________________________________________



def merge_lists(imgs_list, id_ses_list_formated):
    """
    This function merges two lists into one
    """
    imgs_IDSES_tuple = list(zip(imgs_list, id_ses_list_formated))
    return imgs_IDSES_tuple

def split_list(merge_list):
    """
    This function splits a list into two lists
    """
    list_1 = [tuple[0] for tuple in merge_list]
    list_2 = [tuple[1] for tuple in merge_list]
    return  list_1, list_2



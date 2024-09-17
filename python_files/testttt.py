import pandas as pd  
import os
import glob
import re
import yaml



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






#HERE ARE THE NEW FUNCTIONS FOR OBTAINING THE ID AND SESSIONS.



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








def merge_id_ses_to_ADNIMERGE(transformed_id_ses_list, ADNIMERGE_df):
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
    #Drop all columns except ID, Session and ADAS11
    df_ADNI_BIDS_id_ses_ADAS = df_ADNI_BIDS[['PTID', 'VISCODE', 'ADAS11']]
    return df_ADNI_BIDS_id_ses_ADAS

config_file = 'config.yaml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

    folder = config['loader']['load_folder']['folder']
    prefix = config['loader']['load_folder']['prefix']
    extension = config['loader']['load_folder']['extension']
    target_folder_name = config['loader']['load_folder']['target_folder_name']
    
    epochs = config['model']['epochs']    
    device = config['experiment']['device']
    #Load images and normalise them
    imgs_paths = find_pet(folder, prefix, extension, target_folder_name)


tuple_id_ses = extract_id_ses_from_path(imgs_paths)
transformed_tuple_id_ses = transform_string(tuple_id_ses)
    
    
path = 'C:/Users/Cristobal/Desktop/VAE_test/python_files/ADNIMERGE.csv'   
ADNIMERGE_df = pd.read_csv(path)
    

df_ADNI_BIDS_id_ses_ADAS = merge_id_ses_to_ADNIMERGE(transformed_tuple_id_ses, ADNIMERGE_df)

print(df_ADNI_BIDS_id_ses_ADAS)
print(f'Max value ADAS', df_ADNI_BIDS_id_ses_ADAS['ADAS11'].max())
print(f'Min value ADAS', df_ADNI_BIDS_id_ses_ADAS['ADAS11'].min())

import pandas as pd
from load_database import transform_string, extract_id_ses_from_path, merge_id_ses_to_ADNIMERGE, find_pet
import yaml


config_file = 'config.yaml'
    
    #LOAD HYPERPARAMETERS FROM CONFIG FILE

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

    folder = config['loader']['load_folder']['folder']
    prefix = config['loader']['load_folder']['prefix']
    extension = config['loader']['load_folder']['extension']
    target_folder_name = config['loader']['load_folder']['target_folder_name']
    
    epochs = config['model']['epochs']    
    device = config['experiment']['device']
    splits = config['loader']['splits']
    path_ADNIMERGE = config['loader']['load_ADNIMERGE']

imgs_paths = find_pet(folder, prefix, extension, target_folder_name)
id_ses_list = extract_id_ses_from_path(imgs_paths)
id_ses_list_formated = transform_string(id_ses_list)
print(id_ses_list_formated[4]) 



path = 'C:/Users/Cristobal/Desktop/VAE_test/python_files/ADNIMERGE.csv'

df = pd.read_csv(path)

df_id_ses = set(df[['PTID', 'VISCODE']].itertuples(index = False, name= None))


def is_valid_id(id, valid_ids):
    return id in valid_ids


missing_data = [(id, ses) for (id, ses) in id_ses_list_formated if (id, ses) not in df_id_ses] 

print("Missing subject", missing_data)

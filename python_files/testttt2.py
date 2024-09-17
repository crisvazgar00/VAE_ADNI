import os
import glob
import yaml
from load_database import find_pet, load_img_nii, normalization_cerebellum, split_database, transform_string, extract_id_ses_from_path, merge_id_ses_to_ADNIMERGE
import pandas as pd




def merge_lists(imgs_list, id_ses_list_formated):
    imgs_IDSES_tuple = list(zip(imgs_list, id_ses_list_formated))
    return imgs_IDSES_tuple


def split_list(merge_list):
    """
    This function splits a list into two lists
    """
    list_1 = [tuple[0] for tuple in merge_list]
    list_2 = [tuple[1] for tuple in merge_list]
    return  list_1, list_2




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
    
#LOAD IMAGES AND APPLY NORMALIZATION
imgs_paths = find_pet(folder, prefix, extension, target_folder_name)
imgs_list = load_img_nii(imgs_paths)
id_ses_list = extract_id_ses_from_path(imgs_paths)
id_ses_list_formated = transform_string(id_ses_list)

#CREATE A LIST (IMGS, (ID, SES))
imgs_IDSES_tuple = merge_lists(imgs_list, id_ses_list_formated)
paths_imgs_IDSES_tuple = merge_lists(imgs_paths, imgs_IDSES_tuple)
#SHUFFLE AND SPLIT DATABASE


    
train_set, eval_set, test_set = split_database(paths_imgs_IDSES_tuple, splits)



_, train_set = split_list(train_set) #SPLIT INTO (PATH) AND (IMG, (ID, SES))
train_set, _ = split_list(train_set) #SPLIT INTO (IMG) ((ID,SES))

_, eval_set = split_list(train_set)
eval_set, _ = split_list(eval_set)

#WE ONLY NEED (ID, SES) FOR TEST SET
path_test, test_set = split_list(test_set)
test_set, test_id_ses = split_list(test_set) 


#NOW WE NEED TO GET A TUPLE LIST (ID, SES, ADAS13)

df_ADNIMERGE = pd.read_csv(path_ADNIMERGE)
df_ID_SES_ADAS = merge_id_ses_to_ADNIMERGE(test_id_ses, df_ADNIMERGE)

train_set_set = set(train_set)
eval_set_set = set(eval_set)
test_set_set = set(test_set)

intersect_train_eval = train_set_set & eval_set_set
intersect_train_test = train_set_set & test_set_set
intersect_eval_test = eval_set_set & test_set_set


if not intersect_train_eval and not intersect_train_test and not intersect_eval_test:
    print("Todos los elementos en train_set, eval_set y test_set son distintos.")
else:
    if intersect_train_eval:
        print(f"Intersección entre train_set y eval_set: {intersect_train_eval}")
    if intersect_train_test:
        print(f"Intersección entre train_set y test_set: {intersect_train_test}")
    if intersect_eval_test:
        print(f"Intersección entre eval_set y test_set: {intersect_eval_test}")

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from load_database import find_pet, load_img_nii, split_database, transform_string, extract_id_ses_from_path, merge_id_ses_to_ADNIMERGE
from load_database import merge_lists, split_list
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#___________________________________________________________

#           FUNCTIONS FOR PLOTTING ADAS13 RESULTS
#___________________________________________________________



def plot_latent_space_ADAS(z_list, id_ses_list, df_ADNIMERGE, path_test, config):
    """
    Function for plotting latent_dim x against latent_dim for test_set.
    Each point in the scatterplot is colorcoded by its ADAS score.
        Args:
        
            -z_list: List of dimensions (length(test_set), lat_space). Contains
            the latent space of every test_set subject (or whatever we feed the function)
            
            -test_id_ses: List of tuples (ID, SES, ADAS) of each test_set subject. The order
            of z_list matches the order of test_id_ses (they are sorted together)
            
            -df_ADNIMERGE: ADNIMERGE dataframe containing the information of every subject of
            the study (More subjects than our database).
            
            -config: configuration file with hyprparameters. Used to obtain latent_dim.
    """
    
    results_folder = "results"
    sub_folder = os.path.join(results_folder, "scatter_images_ADAS")
    
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    
    latent_dim = config['model']['latent']
    latents_plot = np.arange(latent_dim)
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, ADAS13)
    feature = config['results']['feature_ADAS']
    df_ID_SES_ADAS = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_ADAS = df_ID_SES_ADAS.set_index(['PTID', 'VISCODE'])
    df_ADAS = df_ID_SES_ADAS.loc[df_order.index]
    #GET THE INDEX LIST TO PRINT INTO FILE
    index_ADAS = df_ADAS.index

    ADAS_array = df_ADAS['ADAS13'].values
    print('_______________________________________________________________________________________')
    print(f' IN PLOT_LATENT_SPACE_ADAS: ¿Any NaN value in ADAS_list?: {np.isnan(ADAS_array).any()}')
    
    z_list = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z_list]
    # Crear una lista de tuplas (z, ADAS) para mantener el orden
    paired_data = list(zip(z_list, ADAS_array))

    # Filtrar las tuplas donde el segundo elemento no sea NaN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    z_clean, ADAS_clean = zip(*filtered_data)
    print(f' IN PLOT_LATENT_SPACE_ADAS: ¿Any NaN value in ADAS_clean?: {np.isnan(ADAS_clean).any()}')
    
    ADAS_list_nonzero = np.where(np.array(ADAS_clean) == 0, 1e-10, ADAS_clean)

    
    #FOR DEBUGGING PURPOSES. SAVE INTO FILE PATH_ID AND ADNIMERGE_ID
    debugging = config['experiment']['debugging']
    
    if debugging == True:
        #TO SAVE THE LATENT SPACE OF EACH PATIENT OF TEST
        with open(os.path.join(results_folder, 'latent_space.txt'), 'w') as file:
            for item in z_list:
                file.write(f'{item}\n')
        #TO SAVE THE ID SES LIST ORDER OF THE LATENT SPACE
        with open(os.path.join(results_folder, 'latent_space_ID_SES_order.txt'), 'w') as file:
            for item in id_ses_list:
                file.write(f'{item}\n')
        
        #TO SAVE BOTH PATH OF ID AND ADAS INDEX AND SEE IF THEY MATCH
        path_test = extract_id_ses_from_path(path_test)
        test_path_ID_SES = merge_lists(path_test, id_ses_list)
        test_ADNIMERGE_path_ID_SES = merge_lists(test_path_ID_SES, index_ADAS)
        columns = ["test_path_ADNI_BIDS", "test_ADNIBIDS_ID_SES", "test_ADNIMERGE_ID_SES"]
        with open(os.path.join(results_folder, 'paths_ID_SES.txt'), 'w') as file:
            file.write('\t'.join(columns) + '\n')
            for item in test_ADNIMERGE_path_ID_SES:
                file.write(f'{item}\n')
                
#print(f'len of z_list is: {len(z_list)}, and shape is: {z_list[0].shape}')
        

    for latx in latents_plot:
        for laty in latents_plot:
            if latx != laty:
                dim_x = [item [latx] for item in z_clean] #GET DIM X OF LATENT SPACE
                dim_y = [item [laty] for item in z_clean] #GET DIM Y OF LATENT SPACE

                
                plt.figure(figsize = (8, 6))
                plt.scatter(dim_x, dim_y, c = np.log(ADAS_list_nonzero), cmap='viridis', edgecolors='k')

                plt.xlabel(f'Latent dim {latx}')
                plt.ylabel(f'Latent dim {laty}')
                plt.title (f'Latent dim {latx} vs {laty}')

                plt.colorbar(label='ADAS13')
                plt.savefig(f'{sub_folder}/scatter_plot_dim{latent_dim}_lat{latx}_lat{laty}.png', format='png', dpi=300, bbox_inches='tight')
                plt.close()
    return None



def plot_latentx_vs_ADAS(z_list, id_ses_list, df_ADNIMERGE, path_test, config):
    """
    Function for plotting latent_dim x against latent_dim for test_set.
    Each point in the scatterplot is colorcoded by its ADAS score.
        Args:
        
            -z_list: List of dimensions (length(test_set), lat_space). Contains
            the latent space of every test_set subject (or whatever we feed the function)
            
            -test_id_ses: List of tuples (ID, SES, ADAS) of each test_set subject. The order
            of z_list matches the order of test_id_ses (they are sorted together)
            
            -df_ADNIMERGE: ADNIMERGE dataframe containing the information of every subject of
            the study (More subjects than our database).
            
            -config: configuration file with hyprparameters. Used to obtain latent_dim.
    """
    results_folder = 'results'
    sub_folder = os.path.join(results_folder, "scatter_images_latx_VS_ADAS")
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
        
    
    
    latent_dim = config['model']['latent']
    latents_plot = np.arange(latent_dim)
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, ADAS13)
    feature = config['results']['feature_ADAS']
    df_ID_SES_ADAS = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_ADAS = df_ID_SES_ADAS.set_index(['PTID', 'VISCODE'])
    df_ADAS = df_ID_SES_ADAS.loc[df_order.index]


    ADAS_list = df_ADAS['ADAS13'].values #THIS IS THE SORTED LIST OF ADAS TO USE
    
    z_list = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z_list]
    # Crear una lista de tuplas (z, ADAS) para mantener el orden
    paired_data = list(zip(z_list, ADAS_list))

    # Filtrar las tuplas donde el segundo elemento no sea NaN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    z_clean, ADAS_clean = zip(*filtered_data)
    
    ADAS_list_nonzero = np.where(np.array(ADAS_clean) == 0, 1e-10, ADAS_clean)
    
    paths_test_dir = os.path.join(results_folder, "paths_test_dir_ADAS")

    if not os.path.exists(paths_test_dir):
        os.makedirs(paths_test_dir)
        
    df_ADAS.to_csv(f'{paths_test_dir}/ADAS_CSV.txt', sep=' ', index=True)
        

    
    for latx in latents_plot:
            dim_x = [item [latx] for item in z_clean] #GET DIM X OF LATENT SPACE
                
            plt.figure(figsize = (8, 6))
            plt.scatter(dim_x, ADAS_list_nonzero)

            plt.xlabel(f'Latent dim {latx}')
            plt.ylabel(f'ADAS13')

            plt.savefig(f'{sub_folder}/scatter_plot_dim{latent_dim}_lat{latx}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
    return None



def SVR_ADAS(z, id_ses_list, df_ADNIMERGE, path_test, config):
    results_folder = "results"
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    debugging = config['experiment']['debugging']
    
    latent_dim = config['model']['latent']
    latents_plot = np.arange(latent_dim)
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, ADAS13)
    feature = config['results']['feature_ADAS']
    df_ID_SES_ADAS = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_ADAS = df_ID_SES_ADAS.set_index(['PTID', 'VISCODE'])
    df_ADAS = df_ID_SES_ADAS.loc[df_order.index]
    #GET THE INDEX LIST TO PRINT INTO FILE
    index_ADAS = df_ADAS.index

    ADAS_array = df_ADAS['ADAS13'].values
    print(f' IN SVR_ADAS: ¿Any NaN value in ADAS_list?: {np.isnan(ADAS_array).any()}')
    
    
    z = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z]
    
    # Crear una lista de tuplas (z, ADAS) para mantener el orden
    paired_data = list(zip(z, ADAS_array))

    # Filtrar las tuplas donde el segundo elemento no sea NaN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    
    if debugging == True:
        with open(os.path.join(results_folder, 'z_ADAS_list_filtered.txt'), 'w') as file:
            for item in filtered_data:
                file.write(f'{item}\n')
    
    # Separar los datos filtrados en listas X_clean y y_clean
    X_clean, y_clean = zip(*filtered_data)

    print(f' IN SVR_ADAS: ¿Any NaN value in filtered ADAS_list?: {np.isnan(y_clean).any()}')
    
    print('_______________________________________________________________________________________')
    
    print(f' Maximum of ADAS: {np.max(y_clean)}')
    print(f' Minimum of ADAS: {np.min(y_clean)}')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    model = LinearSVR()
    
    #Train model
    model.fit(X_train, y_train)
    
    #Predict for test set
    y_pred = model.predict(X_test)
    
    #Evaluate model with MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean squared error of test and prediction of ADAS: {mse}')
    print(f'R2 score of test and prediction of ADAS: {r2}')
    print(f'Coefficients of LinearSVR model of ADAS: {model.coef_}')
    
    with open(os.path.join(results_folder, 'MSE_R2_SVR_ADAS.txt'), 'w') as file:
        file.write(f'{mse}, {r2}\n')
            
    with open(os.path.join(results_folder, 'SVR_COEFFS_ADAS.txt'), 'w') as file:
        file.write(f'{model.coef_}\n')
        
        
#______________________________________________________________________

#           FUNCTIONS FOR PLOTTING VENTRICLES RESULTS      
#______________________________________________________________________  



def plot_latentx_vs_VENTRICLES(z_list, id_ses_list, df_ADNIMERGE, path_test, config):
    """
    Function for plotting latent_dim x against latent_dim for test_set.
    Each point in the scatterplot is colorcoded by its VENTRICLE score.
        Args:
        
            -z_list: List of dimensions (length(test_set), lat_space). Contains
            the latent space of every test_set subject (or whatever we feed the function)
            
            -test_id_ses: List of tuples (ID, SES, ADAS) of each test_set subject. The order
            of z_list matches the order of test_id_ses (they are sorted together)
            
            -df_ADNIMERGE: ADNIMERGE dataframe containing the information of every subject of
            the study (More subjects than our database).
            
            -config: configuration file with hyprparameters. Used to obtain latent_dim.
    """
    results_folder = 'results'
    sub_folder = os.path.join(results_folder, "scatter_images_latx_VS_VENTRICLES")
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
        
    
    
    latent_dim = config['model']['latent']
    latents_plot = np.arange(latent_dim)
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, VENTRICLE)
    feature = config['results']['feature_Ventricles']
    df_ID_SES_VENTRICLES = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_VENTRICLES = df_ID_SES_VENTRICLES.set_index(['PTID', 'VISCODE'])
    df_VENTRICLES = df_ID_SES_VENTRICLES.loc[df_order.index]


    VENTRICLES_array = df_VENTRICLES['Ventricles'].values #THIS IS THE SORTED LIST OF ADAS TO USE
    print(f' IN PLOT_LATENTX_VS_VENTRICLES: ¿Any NaN value inVENTRIBLES_list?: {np.isnan(VENTRICLES_array).any()}')
    
    
    z_list = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z_list]
    # Crear una lista de tuplas (z, ADAS) para mantener el orden
    paired_data = list(zip(z_list, VENTRICLES_array))

    # Filtrar las tuplas donde el segundo elemento no sea NaN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    z_clean, VENTRICLES_clean = zip(*filtered_data)
    print(f' IN PLOT_LATENTX_VS_VENTRICLES: ¿Any NaN value in filtered VENTRIBLES_clean?: {np.isnan(VENTRICLES_clean).any()}')
    
    
    paths_test_dir_VENTRICLES = os.path.join(results_folder, "paths_test_dir_VENTRICLES")

    if not os.path.exists(paths_test_dir_VENTRICLES):
        os.makedirs(paths_test_dir_VENTRICLES)
        
    df_VENTRICLES.to_csv(f'{paths_test_dir_VENTRICLES}/VENTRICLES_CSV.txt', sep=' ', index=True)
        

    
    for latx in latents_plot:
            dim_x = [item [latx] for item in z_clean] #GET DIM X OF LATENT SPACE
                
            plt.figure(figsize = (8, 6))
            plt.scatter(dim_x, np.log(VENTRICLES_clean))

            plt.xlabel(f'Latent dim {latx}')
            plt.ylabel(f'Ventricles')

            plt.savefig(f'{sub_folder}/scatter_plot_dim{latent_dim}_lat{latx}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
    return None






def plot_latent_space_VENTRICLES(z_list, id_ses_list, df_ADNIMERGE, path_test, config):
    """
    Function for plotting latent_dim x against latent_dim for test_set.
    Each point in the scatterplot is colorcoded by its ADAS score.
        Args:
        
            -z_list: List of dimensions (length(test_set), lat_space). Contains
            the latent space of every test_set subject (or whatever we feed the function)
            
            -test_id_ses: List of tuples (ID, SES, ADAS) of each test_set subject. The order
            of z_list matches the order of test_id_ses (they are sorted together)
            
            -df_ADNIMERGE: ADNIMERGE dataframe containing the information of every subject of
            the study (More subjects than our database).
            
            -config: configuration file with hyprparameters. Used to obtain latent_dim.
    """
    results_folder = 'results'
    sub_folder = os.path.join(results_folder, "scatter_images_VENTRICLES")
    
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    
    latent_dim = config['model']['latent']
    latents_plot = np.arange(latent_dim)
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, VENTRICLES)
    feature = config['results']['feature_Ventricles']
    df_ID_SES_VENTRICLES = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_VENTRICLES = df_ID_SES_VENTRICLES.set_index(['PTID', 'VISCODE'])
    df_VENTRICLES = df_ID_SES_VENTRICLES.loc[df_order.index]
    #GET THE INDEX LIST TO PRINT INTO FILE
    index_VENTRICLES = df_VENTRICLES.index

    VENTRICLES_list = df_VENTRICLES['Ventricles'].values

    z_list = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z_list]
    # Crear una lista de tuplas (z, ADAS) para mantener el orden
    paired_data = list(zip(z_list, VENTRICLES_list))

    # Filtrar las tuplas donde el segundo elemento no sea NaN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    z_clean, VENTRICLES_clean = zip(*filtered_data)
    
    
    #FOR DEBUGGING PURPOSES. SAVE INTO FILE PATH_ID AND ADNIMERGE_ID
    debugging = config['experiment']['debugging']
    
    if debugging == True:
        
        with open(os.path.join(results_folder, 'IDX_order_ADNI_BIDS.txt'), 'w') as file:
            for item in id_ses_list:
                file.write(f'{item}\n')
        
        
        path_test = extract_id_ses_from_path(path_test)
        test_path_ID_SES = merge_lists(path_test, id_ses_list)
        test_ADNIMERGE_path_ID_SES = merge_lists(test_path_ID_SES, index_VENTRICLES)
        columns = ["test_path_ADNI_BIDS", "test_ADNIBIDS_ID_SES", "test_ADNIMERGE_ID_SES"]
        with open(os.path.join(results_folder, 'paths_ID_SES.txt'), 'w') as file:
            file.write('\t'.join(columns) + '\n')
            for item in test_ADNIMERGE_path_ID_SES:
                file.write(f'{item}\n')
        

    for latx in latents_plot:
        for laty in latents_plot:
            if latx != laty:
                dim_x = [item [latx] for item in z_clean] #GET DIM X OF LATENT SPACE
                dim_y = [item [laty] for item in z_clean] #GET DIM Y OF LATENT SPACE

                
                plt.figure(figsize = (8, 6))
                plt.scatter(dim_x, dim_y, c = np.log(VENTRICLES_clean), cmap='viridis', edgecolors='k')

                plt.xlabel(f'Latent dim {latx}')
                plt.ylabel(f'Latent dim {laty}')
                plt.title (f'Latent dim {latx} vs {laty}')

                plt.colorbar(label='Ventricles')
                plt.savefig(f'{sub_folder}/scatter_plot_dim{latent_dim}_lat{latx}_lat{laty}.png', format='png', dpi=300, bbox_inches='tight')
                plt.close()
    return None




def SVR_VENTRICLES(z, id_ses_list, df_ADNIMERGE, path_test, config):
    results_folder = "results"
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    debugging = config['experiment']['debugging']
    
    latent_dim = config['model']['latent']
    latents_plot = np.arange(latent_dim)
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, VENTRICLES)
    feature = config['results']['feature_Ventricles']
    df_ID_SES_VENTRICLES = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_VENTRICLES = df_ID_SES_VENTRICLES.set_index(['PTID', 'VISCODE'])
    df_VENTRICLES = df_ID_SES_VENTRICLES.loc[df_order.index]
    #GET THE INDEX LIST TO PRINT INTO FILE
    index_VENTRICLES = df_VENTRICLES.index

    VENTRICLES_array = df_VENTRICLES['Ventricles'].values
    print(f' IN SVR_VENTRICLES: ¿Any NaN value in VENTRICLES_list?: {np.isnan(VENTRICLES_array).any()}')
    
    
    z = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z]
    
    # Crear una lista de tuplas (z, VENTRICLES) para mantener el orden
    paired_data = list(zip(z, VENTRICLES_array))

    # Filtrar las tuplas donde el segundo elemento no sea NaN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    
    if debugging == True:
        with open(os.path.join(results_folder, 'z_VENTRICLES_list_filtered.txt'), 'w') as file:
            for item in filtered_data:
                file.write(f'{item}\n')
    
    # Separar los datos filtrados en listas X_clean y y_clean
    X_clean, y_clean = zip(*filtered_data)

    print(f' IN SVR_VENTRICLES: ¿Any NaN value in filtered VENTRICLES_list?: {np.isnan(y_clean).any()}')
    
    print('_______________________________________________________________________________________')
    
    print(f' Maximum of VENTRICLES: {np.max(y_clean)}')
    print(f' Minimum of VENTRICLES: {np.min(y_clean)}')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    model = LinearSVR()
    
    #Train model
    model.fit(X_train, y_train)
    
    #Predict for test set
    y_pred = model.predict(X_test)
    
    #Evaluate model with MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean squared error of test and prediction for VENTRICLES: {mse}')
    print(f'R2 score of test and prediction for VENTRICLES: {r2}')
    print(f'Coefficients of LinearSVR model for VENTRICLES: {model.coef_}')
    
    with open(os.path.join(results_folder, 'MSE_R2_SVR_VENTRICLES.txt'), 'w') as file:
        file.write(f'{mse}, {r2}\n')
            
    with open(os.path.join(results_folder, 'SVR_COEFFS_VENTRICLES.txt'), 'w') as file:
        file.write(f'{model.coef_}\n')
        
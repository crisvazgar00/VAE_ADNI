import numpy as np
import torch
import os
import nibabel as nib
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def make_grid_recon(batch, string):
    
    slice_index = 45 #SLICE OF VOLUME
    
    
    batch_slices = batch[:, :, :, :, slice_index] #[BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
    
    batch_size = batch_slices.size(0)
    nrow = int(np.sqrt(batch_size))

    
    grid = make_grid(batch_slices, nrow = nrow)
    
    result_folder = 'results'
    os.makedirs(result_folder, exist_ok=True)
    
    grid_normalized = (grid - grid.min()) / (grid.max() - grid.min())
    
    file_path = os.path.join(result_folder, f'grid_images{string}.png')
    plt.imsave(file_path, grid_normalized.permute(1, 2, 0).cpu().numpy())
    return None


def tensor_to_nii(x_input, x_recon):
    """
    This function transforms an input volume tensor 'x_input' into a .nii
    volume 'x_recon'
    """
    
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    x_recon = x_recon.squeeze(0) #Erase channel dimension
    x_numpy = x_recon.cpu().detach().numpy()
    x_nifti_recon = nib.Nifti1Image(x_numpy, affine = np.eye(4))
    
    x_input = x_input.squeeze(0)
    x_input = x_input.cpu().detach().numpy()
    x_nifti_input = nib.Nifti1Image(x_input, affine = np.eye(4))
    nib.save(x_nifti_recon, os.path.join(results_folder, 'reconstructed_test_volume.nii'))
    nib.save(x_nifti_input, os.path.join(results_folder, 'normalised_test_volume.nii'))
    return 



def reconstruction_diff(x, y):
    """
    This function measures the reconstruction difference between two volumes.
    Difference is performed by direct substraction of pixels.
    """
    
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    x = x.squeeze(0) #Erase channel dimension (We need 3dims)
    y = y.squeeze(0) #Erase channel dimension (We need 3dims)
    recon_img = x - y
    recon_img_np = recon_img.detach().cpu().numpy()
    nii_recon_diff = nib.Nifti1Image(recon_img_np, affine = np.eye(4))
    nib.save(nii_recon_diff, os.path.join(results_folder, 'ImageDiff.nii'))
    return recon_img
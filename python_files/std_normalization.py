#%% 
import nibabel as nib 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from skimage import filters

dataset = pd.read_csv('ADNI_BIDS/participants_adnimerge.csv')
dataset_pet = dataset.loc[dataset.Modality=='PET']
BASE_PATH = 'ADNI_BIDS'

#%% 
immean = np.zeros((128,128,90))
N_FILES = len(dataset_pet)
for index, imname in tqdm(dataset_pet.iterrows(), total=len(dataset_pet)): 
    img = nib.load(imname.out_path)
    immean += img.get_fdata().squeeze()/N_FILES

img_mean = nib.Nifti1Image(immean, affine=img.affine, header=img.header, extra=img.extra)
nib.save(img_mean, BASE_PATH+'adni_mean.nii.gz')
norm = filters.threshold_otsu(immean.flatten())
bg_im = nib.Nifti1Image((immean<norm).astype(int), affine=img.affine, header=img.header, extra=img.extra)
nib.save(bg_im, BASE_PATH+'adni_bg.nii.gz')

#%% Image STD: 
from scipy.ndimage import binary_erosion, binary_dilation
imstd = np.zeros((91, 109, 91))
N_FILES = len(dataset_pet)
for index, imname in tqdm(dataset_pet.iterrows(), total=len(dataset_pet)): 
    img = nib.load(imname.out_path)
    imstd += (img.get_fdata().squeeze()-immean.get_fdata().squeeze())**2/(N_FILES-1)
imstd /= imstd.max()
img_std = nib.Nifti1Image(imstd, affine=img.affine, header=img.header, extra=img.extra)
#%% 
nib.save(img_std, BASE_PATH+'ppmi_std.nii.gz')
imns = 1.0*binary_dilation(binary_erosion((imstd<.3), np.ones((3,3,3))), np.ones((3,3,3)))
img_ns = nib.Nifti1Image(imns, affine=img.affine, header=img.header, extra=img.extra)
nib.save(img_ns, BASE_PATH+'ppmi_ns.nii.gz')
# %%

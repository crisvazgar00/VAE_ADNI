import torch
import numpy as np
import os
from VAE_model import VAE, VAE_encoder, VAE_decoder
from debugging import tensor_to_nii, make_grid_recon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

results_folder = 'results'

#LOAD LATENT_DIM TO FEED TO VAE MODEL
checkpoint = torch.load('vae_model.pth')
latent_dim = checkpoint['latent_dim']
model = VAE(encoder=VAE_encoder(latent = latent_dim), decoder=VAE_decoder(latent = latent_dim))

#LOAD WEIGHTS OF LAST SAVED MODEL
model.load_state_dict(checkpoint['model_state_dict'])

#LOAD OPTIMIZER AND UPDATE ITS STATE
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss_eval = checkpoint['loss_eval']
loss_eval_div = checkpoint['loss_eval_div']
loss_eval_recon = checkpoint['loss_eval_recon']
model.eval()

print(f'Model loaded. Last Epoch: {epoch}, Total loss of evaluation: {loss_eval}, Divergence loss: {loss_eval_div}, Loss of recon: {loss_eval_recon}.')

#CHOOSE LATENT VARIABLES
latx = 4
laty = 1

latent_values = np.linspace(-3, 3, 6)
recon_img_list = []
recon_slice_list = []

for value1 in latent_values:
    for value2 in latent_values:
        latent_space = torch.zeros(latent_dim)
        latent_space[latx] = value1
        latent_space[laty] = value2

        with torch.no_grad():
            recon_img = model.decoder(latent_space.unsqueeze(0))
        recon_img_list.append(recon_img.squeeze().numpy())

for img in recon_img_list:
    slice = img[:,:,45]
    recon_slice_list.append(slice)


fig, axs = plt.subplots(len(latent_values), len(latent_values), figsize=(20, 20), gridspec_kw={'wspace': 0, 'hspace': 0})  

fig.text(0.5, 0.02, f'Latent {latx}', ha='center', fontsize = 26)
fig.text(0.02, 0.5, f'Latent {laty}', va='center', rotation='vertical', fontsize = 26)

for i in range(len(latent_values)):
    for j in range(len(latent_values)):
        axs[i, j].imshow(recon_slice_list[i * len(latent_values) + j], cmap='gray', aspect = 'equal')  
        axs[i, j].axis('off')
        
        axs[i, j].set_xlabel(f'{latx}') 
        axs[i, j].set_ylabel(f'{laty}') 
        

plt.subplots_adjust(hspace=0.0)
plt.savefig(os.path.join(results_folder, f'grid_generate_manifold_lat{latx}_lat{laty}.png'), bbox_inches='tight', dpi=300)  


plt.show()
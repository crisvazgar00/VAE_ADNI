import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import nibabel as nib
from typing import Tuple, Union






def SSIM_3D(x: Tensor, y: Tensor, window_size: int = 5, reduction: str = 'mean', window_aggregation: str = 'mean') -> Union[Tensor, Tuple[Tensor, int]]:
    """ 
    Derived from: https://stackoverflow.com/questions/71357619/how-do-i-compute-batched-sample-covariance-in-pytorch
    Computes the Structural Similarity Index Measure (SSIM) for 3D images.

    Parameters:
    x (Tensor): A tensor representing the first batch of 3D images.
    y (Tensor): A tensor representing the second batch of 3D images.
    window_size (int): The size of the window to consider when computing the SSIM. Default is 5.
    reduction (str): The type of reduction to apply to the output: 'mean' | 'sum'. Default is 'mean'.
    window_aggregation (str): The type of aggregation to apply to the window: 'mean' | 'sum'. Default is 'mean'.

    Returns:
    Tensor: The SSIM value.
    int: The number of patches if reduction is 'sum'. Otherwise, returns None.
    """
    # Convert images to float and create patches
    patched_x = x.to(dtype=torch.float32).unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .reshape(x.shape[0], -1, window_size**3)
    patched_y = y.to(dtype=torch.float32).unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .reshape(y.shape[0], -1, window_size**3)

    # Compute statistics
    B, P, D = patched_x.size()
    varx, mux = torch.var_mean(patched_x, dim=-1)
    vary, muy = torch.var_mean(patched_y, dim=-1)
    diffx = (patched_x - mux.unsqueeze(-1)).reshape((B*P, -1))
    diffy = (patched_y - muy.unsqueeze(-1)).reshape((B*P, -1))
    covs = torch.bmm(diffx.unsqueeze(1), diffy.unsqueeze(2)).squeeze().reshape(B, P)/(D-1)

    # Compute SSIM
    c1, c2 = 0.01, 0.03
    numerador = (2*mux*muy + c1)*(2*covs + c2)
    denominador = (mux**2 + muy**2 + c1)*(varx + vary + c2)
    if window_aggregation == 'sum':
        ssim_bp = (numerador/denominador).sum(dim=-1)  # sum over windows
    elif window_aggregation == 'mean':
        ssim_bp = (numerador/denominador).mean(dim=-1)
    else:
        print(f'window reduction {window_aggregation} not supported')
        return None, None
    if reduction=='sum':
        raise ValueError(f'Window aggregation {window_aggregation} not supported')
    if reduction == 'sum':
        return ssim_bp.sum(), P
    else:
        return ssim_bp.mean(), None
    
def DSSIM_3D(x, y, window_size=5, reduction='mean', window_aggregation='mean'):
    ssim, P = SSIM_3D(x, y, window_size=window_size, reduction=reduction, window_aggregation=window_aggregation)
    if window_aggregation=='mean':
        P=1
    return (P-ssim)






class VAE_encoder(nn.Module):
    """
    Class for a 3D Encoder
    """
    
    def __init__(self, latent:int=10):
        """
        We define 2 convolutional layers, 1 fully connected layer
        and finally a fully connected layer for both mean and logvar
            Arg:
                - latent: dimension of the latent space
        We do not use pooling layers because stride=2 reduces the dimension
        """
        
        super().__init__()
        self.latent = latent
        self.conv1 = nn.Conv3d(1 , 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv3d(32 , 64, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv3d(64 , 128, kernel_size = 3, stride = 2, padding = 1)
        #Our input images have shape (90, 110, 90) so output shape is (12, 14, 12)
        self.fc = nn.Linear(128 * 12 * 14 * 12  , 256) # Feat_maps * H * W * D (401.408)
        self.lat_mean = nn.Linear(256, self.latent)
        self.lat_logvar = nn.Linear(256, self.latent)
        
        
    def reparameterize(self, z_mean, z_logvar):
        """
        This function performs the reparameterization trick.
            Args:
                - z_mean: mean latent space tensor
                - z_logvar: logvar latent space tensor
                
            Output: z: reparameterized sample 
        """
        
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + std*eps
        return z
    
        
    def forward(self, x):
        
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)        
        if len(x.shape) == 4:
            x = x.unsqueeze(1) #Reshapes x into [batch_size, 1, depth, height, width]
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 12 * 14 * 12 ) #flatten data to fit linear layer
        x = F.relu(self.fc(x))
        z_mean = self.lat_mean(x)
        z_logvar = self.lat_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar
    
    
    def divergence_loss(self, z_mean, z_logvar, batch_size):
        """
        Function for computing the KL divergence between
        two gaussians distributions. We assume priori distribution
        to be gaussian.
            Args:
                -z_mean: mean vector of latent space
                -z_logvar: logvar vector of latent space
                -beta: scalar to adjust the effect of divergence (Beta-VAE)
                
            Output:
                -KL_div_loss: KL divergence loss
        
        """
        
        KL_div_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        KL_div_loss /= batch_size
        return KL_div_loss
    
    
    
    
    
class VAE_decoder(nn.Module): 
    """
    Class for 3D VAE Decoder
    """
    
    def __init__(self, latent:int=10):
        super().__init__()
        self.latent = latent
        self.fc = nn.Linear(self.latent, 128 * 12 * 14 * 12)
        # out_padding to fit reconstructed dimensiones to real dimensions. Computed using W_out formula for Trans conv.
        self.conv1Trans = nn.ConvTranspose3d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = (0, 1, 0)) 
        self.conv2Trans = nn.ConvTranspose3d(64, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = (0, 0, 0))
        self.conv3Trans = nn.ConvTranspose3d(32 , 1, kernel_size = 3, stride = 2, padding = 1, output_padding = (1, 1, 1))
        
        
    def forward(self, x):
        #print(f'Hola, soy forward de VAE_decoder y funciono!')
        x = F.relu(self.fc(x))
        x = x.view(-1, 128, 12, 14, 12) #This rearranges shapes in volumes
        x = F.relu(self.conv1Trans(x))
        x = F.relu(self.conv2Trans(x))
        x = F.sigmoid(self.conv3Trans(x))
        print(x.shape)
        return x
    

    def loss_recon(self, x_target, x_recon):
        """
        Function for computing the loss due to reconstruction of 3D volumes.
        We use DSSIM defined as 1-SSIM.
            Args:
                -x: input data. (3D volumes)
                -x_recon: reconstructed data. (3D volumes)
                
                
            Output:
                -loss: reconstruction loss computed with DSSIM
        
        """
        
        recon_loss = DSSIM_3D(x_target, x_recon)
        return recon_loss
    
    
    
    
    
class VAE(nn.Module):
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    
    def forward(self, x):
        z, z_mean, z_logvar = self.encoder(x)
        
        x_recon = self.decoder(z)
        
        return x_recon, z, z_mean, z_logvar
    
    
    def loss(self, x_target, x_recon, z_mean, z_logvar, beta, batch_size):
        
        div_loss = beta*self.encoder.divergence_loss(z_mean, z_logvar, batch_size)
        recon_loss = self.decoder.loss_recon(x_target, x_recon)
        total_loss = recon_loss - beta*div_loss
        
        return  total_loss, div_loss, recon_loss
        

    

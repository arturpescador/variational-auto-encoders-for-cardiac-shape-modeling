import tensorflow as tf
import torch
import torch.nn as nn 
import torch.nn.functional as F

class VAE(nn.Module):

    def __init__(self, n_rows, n_cols, n_channels):
        super(VAE, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels    
        self.z_dim = 32
        self.n_blocks = 4

        self.fc_input_channels = self.n_rows // 2 ** (self.n_blocks + 1)
        self.block_channels = 384
        
        # Encoder
        self.e11 = nn.Conv2d(in_channels=self.n_channels, out_channels=48, kernel_size=3, stride=2)
        self.e12 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding='same')
        
        self.e21 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=2)
        self.e22 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same')
        
        self.e31 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=2)
        self.e32 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding='same')
        
        self.e41 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=2)
        self.e42 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding='same')

        # Latent space layers
        self.fc1 = nn.Linear(self.block_channels*self.fc_input_channels**2, self.z_dim) # fc1 is the mu layer
        self.fc2 = nn.Linear(self.block_channels*self.fc_input_channels**2, self.z_dim) # fc2 is the logvariance layer

        # Decoder
        self.d0 = nn.Linear(self.z_dim, self.block_channels*self.fc_input_channels**2)

        self.d11 = nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=2, stride=2)
        self.d12 = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=3, stride=1, padding='same')

        self.d21 = nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=2, stride=2)
        self.d22 = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3, stride=1, padding='same')

        self.d31 = nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=2, stride=2)
        self.d32 = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding='same')

        self.d41 = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=2, stride=2)
        self.d42 = nn.ConvTranspose2d(in_channels=48, out_channels=self.n_channels, kernel_size=3, stride=1, padding='same')


    def encoder(self, x):
        h = F.relu(self.e11(x))
        h = F.relu(self.e12(h))
        h = F.relu(self.e21(h))
        h = F.relu(self.e22(h))
        h = F.relu(self.e31(h))
        h = F.relu(self.e32(h))
        h = F.relu(self.e41(h))
        h = F.relu(self.e42(h))
        h = h.view(h.size(0), -1)
        return self.fc1(h), self.fc2(h) # return mu, logvariance
    
    def sampling(self, mu, log_var):
        # this function samples a Gaussian distribution, with average (mu) and standard deviation specified (using log_var)
        std = torch.sqrt(torch.exp2( log_var )) 
        eps = torch.randn(self.z_dim) 
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        h = F.relu(self.d0(z))
        h = h.view(-1, self.block_channels, self.fc_input_channels, self.fc_input_channels)
        h = F.relu(self.d11(h))
        h = F.relu(self.d12(h))
        h = F.relu(self.d21(h))
        h = F.relu(self.d22(h))
        h = F.relu(self.d31(h))
        h = F.relu(self.d32(h))
        h = F.relu(self.d41(h))
        h = F.relu(self.d42(h))
        return F.softmax(h, dim=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        `x` : tensor, the input image.
        
        Returns
        -------
        `x_hat` : tensor, the reconstructed image
        `mu` : tensor, the mean of the latent space
        `log_var` : tensor, the log variance of the latent space
        """
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def soft_dice_loss(self, y_true, y_pred):
        """
        Calculate the soft Dice loss between the ground truth and predicted masks.

        Parameters
        ----------
        `y_true` : tensor, the ground truth mask.        
        `y_pred` : tensor, the predicted mask.
        """
        smooth = 1.0
        axes =  tuple(range(1, len(y_pred.shape)-1)) # skip batch and class axis when summing
        intersection = torch.sum( y_pred * y_true, dim=axes ) 
        card_ground_truth = torch.sum( y_true, dim=axes )
        card_predicted = torch.sum( y_pred, dim=axes )
        dice_coeff = (2.0 * intersection + smooth) / (card_ground_truth + card_predicted + smooth) # computed soft dice per sample per class

        loss = 1.0 - torch.mean(dice_coeff) # mean of dice coefficients for all samples and classes
        return loss
    
    def loss_function(self, y_true, y_pred, mu, log_var):
        """
        Calculate the loss function for the VAE.

        Parameters
        ----------
        `y_true` : tensor, the ground truth mask.
        `y_pred` : tensor, the predicted mask.
        `mu` : tensor, the mean of the latent space.
        `log_var` : tensor, the log variance of the latent space.

        Returns
        -------
        `loss` : tensor, the loss value.
        """
        reconstruction_error = self.soft_dice_loss(y_true, y_pred)
        KLD = torch.subtract( torch.add( torch.exp(log_var), torch.pow(mu, 2) ), torch.add(log_var, 1) )/2
        return torch.sum( reconstruction_error ) + torch.sum(KLD)

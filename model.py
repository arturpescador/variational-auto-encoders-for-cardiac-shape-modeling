import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=x_dim, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=x_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # Latent space layers
        self.fc_mu = nn.Linear(32 * 7 * 7, z_dim)
        self.fc_log_var = nn.Linear(32 * 7 * 7, z_dim)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 32, 7, 7)  # Reshape
        return self.decoder(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        `x` : tensor, the input image.
        
        Returns
        -------
        `x_hat` : tensor, the reconstructed image.
        """
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def soft_dice_loss(self, y_pred, y_true):
        """
        Calculate the soft Dice loss between the ground truth and predicted masks.

        Parameters
        ----------
        `y_true` : tensor, the ground truth mask.        
        `y_pred` : tensor, the predicted mask.
        """
        smooth = 1.0
        intersection = torch.sum(y_pred * y_true)
        dice_coeff = (2.0 * intersection + smooth) / (torch.sum(y_pred) + torch.sum(y_true) + smooth)
        loss = 1.0 - dice_coeff
        return loss

    def loss_function(self, y_pred, y_true, mu, log_var):
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
        reconstruction_error = self.soft_dice_loss(y_pred, y_true)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return (reconstruction_error + KLD) / y_pred.size(0)

## Architecture
# We will use the following convoluional architecture for the classifier:
# 
# - conv2d, filter size  3×3 , 32 filters, stride=(2,2), padding="SAME"
# - ReLU
# - conv2d, filter size  3×3 , 32 filters, stride=(2,2), padding="SAME"
# - ReLU
# - MaxPool2D, stride=(2,2)
# - Flatten
# - Dense layer

"""
def class_model_vao(n_epochs, batch_size, learning_rate, nb_classes):

    # number of convolutional filters to use
    nb_filters1 = 16
    nb_filters2 = 32
    nb_filters3 = 64
    nb_filters4 = 96
    # convolution kernel size
    kernel_size_3 = (3, 3)
    kernel_size_4 = (4, 4)
    # size of pooling area for max pooling
    stride_size1 = (1, 1)
    stride_size2 = (2, 2)

    # --- Size of the successive layers
    n_h_0 = 1 #greyscale input images
    n_h_1 = nb_filters1
    n_h_2 = nb_filters2
    n_h_3 = nb_filters3
    n_h_4 = nb_filters4

#i dont know what we must put instead of conv2d, maxpool2d (maybe its ok but idk)
    classification_model = nn.Sequential(nn.Conv2d(n_h_0, n_h_1, kernel_size=kernel_size_3, stride=stride_size2, padding='same'),
                                                    nn.ReLU(),
                                                    nn.Conv2d(n_h_1, n_h_2, kernel_size=kernel_size_3, stride=stride_size2, padding='same'),
                                                    nn.ReLU(),
                                                    nn.Conv2d(n_h_2, n_h_3, kernel_size=kernel_size_3, stride=stride_size2, padding='same'),
                                                    nn.ReLU(),
                                                    nn.Flatten(),
                                                    nn.Linear(n_h_2*(imgs_generated.shape[2]-1)*(imgs_generated.shape[3]-1), nb_classes)) # FILL IN CODE HERE

    criterion = nn.soft_dice_loss() # FILL IN CODE HERE
    optimizer = torch.optim.Adam(classification_model.parameters(), lr=learning_rate)"""
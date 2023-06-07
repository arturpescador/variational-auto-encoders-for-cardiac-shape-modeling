import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):

    """ 
    # Architecture from the article
    def __init__(self, n_rows, n_cols, n_channels):
        super(VAE, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels    
        self.z_dim = 64
        
        # Encoder
        self.e11 = nn.Conv2d(in_channels=self.n_channels, out_channels=16, kernel_size=3, stride=2, padding='same')
        self.e12 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.e21 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding='same')
        self.e22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.e31 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding='same')
        self.e32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.e41 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding='same')


        # Decoder
        self.d11 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=4, stride=2, padding='same')
        self.d12 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same')
        self.d21 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=4, stride=2, padding='same')
        self.d22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.d31 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding='same')
        self.d32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.d41 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding='same')
        self.d42 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same')

        # Latent space layers
        self.fc1 = nn.Linear(7*7*16, self.z_dim) # fc1 is the mu layer
        self.fc2 = nn.Linear(7*7*16, self.z_dim) # fc2 is the logvariance layer

    def encode(self, x):
        h = F.relu(self.e11(x))
        h = F.relu(self.e12(h))
        h = F.relu(self.e21(h))
        h = F.relu(self.e22(h))
        h = F.relu(self.e31(h))
        h = F.relu(self.e32(h))
        h = F.relu(self.e41(h))
        h = h.view(h.size(0), -1)
        return self.fc1(h), self.fc2(h)

    def decode(self, z):
        h = F.relu(self.d11(z))
        h = F.relu(self.d12(h))
        h = F.relu(self.d21(h))
        h = F.relu(self.d22(z))
        h = F.relu(self.d31(h))
        h = F.relu(self.d32(h))
        h = F.relu(self.d41(h))
        h = F.relu(self.d42(h))
        h = h.view(h.size(0), 1, 28, 28)
        return torch.sigmoid(h) # the activation function of the output layer is sigmoid
    """
    def __init__(self, n_rows, n_cols, n_channels):
        super(VAE, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels    
        self.z_dim = 64
        
        # Encoder
        self.e11 = nn.Conv2d(in_channels=self.n_channels, out_channels=16, kernel_size=3, stride=2)
        self.e12 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1)

        # Decoder
        self.d0 = nn.Linear(self.z_dim, 7*7*1)
        self.d11 = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.d12 = nn.ConvTranspose2d(in_channels=16, out_channels=self.n_channels, kernel_size=3, stride=2)

        # Latent space layers
        self.fc1 = nn.Linear(125*125*1, self.z_dim) # fc1 is the mu layer
        self.fc2 = nn.Linear(125*125*1, self.z_dim) # fc2 is the logvariance layer

    def encoder(self, x):
        h = F.relu(self.e11(x))
        h = F.relu(self.e12(h))
        print(h.shape)
        h = h.view(h.size(0), -1)
        print(h.shape)
        return self.fc1(h), self.fc2(h)

    def decoder(self, z):
        h = F.relu(self.d0(z))
        h = h.view(h.size(0), 1, 125, 125)
        h = F.relu(self.d11(h))
        h = F.relu(self.d12(h))
        return nn.LogSoftmax(h, dim=1)
    
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
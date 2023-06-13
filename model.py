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
        
        # Encoder (convolutional part)
        self.e11 = nn.Conv2d(in_channels=self.n_channels, out_channels=48, kernel_size=2, stride=2)
        self.e12 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1) 
        
        self.e21 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=2, stride=2)
        self.e22 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1) # since kernel size is 3, padding of 1 corresponds to 'same'
        
        self.e31 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=2, stride=2)
        self.e32 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        
        self.e41 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=2, stride=2)
        self.e42 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        
        # Pass a meta tensor through the convolutional layers to determine output size: (batch_size, n_channels, n_rows, n_cols)
        # self.fc_input_size = n_rows = n_cols
        # self.fc_input_channels = n_channels
        self.fc_input_channels, self.fc_input_size = self.compute_fc_input_dim()

        # Latent space layers
        self.fc1 = nn.Linear((self.fc_input_size**2)*self.fc_input_channels, self.z_dim) # fc1 is the mu layer
        self.fc2 = nn.Linear((self.fc_input_size**2)*self.fc_input_channels, self.z_dim) # fc2 is the logvariance layer

        # Decoder
        self.d0 = nn.Linear(self.z_dim, (self.fc_input_size**2)*self.fc_input_channels)

        self.d11 = nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=2, stride=2)
        self.d12 = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.d21 = nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=2, stride=2)
        self.d22 = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3, stride=1, padding=1)

        self.d31 = nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=2, stride=2)
        self.d32 = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1)

        self.d41 = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=2, stride=2)
        self.d42 = nn.ConvTranspose2d(in_channels=48, out_channels=self.n_channels, kernel_size=3, stride=1, padding=1)


    def convolutional_encoder(self, x):
        """
        Convolution part of the encoder block.

        Parameters:
        -----------
        `x` : tensor, the input image

        Returns:
        --------
        `h` : tensor, the output of the convolutional encoder block
        """

        layers = [self.e11, self.e12, self.e21, self.e22, self.e31, self.e32, self.e41, self.e42]
        h = x.clone()
        for layer in layers:
            h = F.elu(layer(h))

        return h
    
    def compute_fc_input_dim(self):
        """
        Computes the input size of the fully connected layer of the encoder block.

        Returns:
        --------
        `output.size(1)` : int, the number of channels of the output of the encoder
        `output.size(2)` : int, the number of rows (and of columns) of the output of the encoder
        """

        # a meta tensor has no data
        tensor = torch.zeros(1, self.n_channels, self.n_rows, self.n_cols, device="meta")

        # the tensor is passed through the convolutional layers to determine output size
        output = self.convolutional_encoder(tensor)

        return output.size(1), output.size(2)

    def encoder(self, x):
        """
        Encoder block.
        
        Parameters:
        -----------
        `x` : tensor, the input image.

        Returns:
        --------
        `mu` : tensor, the mean of the latent space
        `logvar` : tensor, the log variance of the latent space
        """

        h = self.convolutional_encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc1(h), self.fc2(h) # return mu, logvariance
    
    def sampling(self, mu, log_var):
        """
        Samples a Gaussian distribution, with average (mu) and standard deviation specified using log_var.

        Parameters:
        -----------
        `mu` : tensor, the mean of the latent space
        `log_var` : tensor, the log variance of the latent space

        Returns:   
        --------
        `z` : tensor, the sampled latent space
        """
        std = torch.sqrt(torch.exp2( log_var )) 
        eps = torch.randn(self.z_dim).to(mu.get_device())
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        """
        Decoder block.
        
        Paramerters:
        ------------
        `z` : tensor, the sampled latent space.

        Returns:
        --------
        `x_hat` : tensor, the reconstructed image.
        """
        h = F.relu(self.d0(z))
        h = h.view(-1, self.fc_input_channels, self.fc_input_size, self.fc_input_size)

        layers = [self.d11, self.d12, self.d21, self.d22, self.d31, self.d32, self.d41, self.d42]
        for layer in layers:
            h = F.elu(layer(h))

        return F.softmax(h, dim=1)
    
    def forward(self, x, test=False):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        `x` : tensor, the input image.
        `test` : bool, whether to use the network in test mode or not.
        
        Returns
        -------
        `x_hat` : tensor, the reconstructed image
        `mu` : tensor, the mean of the latent space
        `log_var` : tensor, the log variance of the latent space
        """
        mu, log_var = self.encoder(x)
        if test:
            z = mu
        else:
            z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def soft_dice_loss(self, y_true, y_pred, reduction="sum"):
        """
        Calculate the soft Dice loss between the ground truth and predicted masks.

        Parameters
        ----------
        `y_true` : tensor, the ground truth mask.        
        `y_pred` : tensor, the predicted mask.
        `reduction` : str, the reduction method to use. Can be "sum" or "mean".
        """
        smooth = 1e-5
        axes =  tuple(range(1, len(y_pred.shape)-1)) # skip batch and class axis when summing
        intersection = torch.sum( y_pred * y_true, dim=axes ) 
        card_ground_truth = torch.sum( y_true, dim=axes )
        card_predicted = torch.sum( y_pred, dim=axes )
        dice_coeff = 1 - (2.0 * intersection + smooth) / (card_ground_truth + card_predicted + smooth) # computed soft dice per sample per class

        if reduction == "sum":
            loss = torch.sum(dice_coeff)
        elif reduction == "mean":
            loss = torch.mean(dice_coeff)
        else:
            raise ValueError("reduction must be either sum or mean")
        
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

    def train_one_epoch(self, optimizer, data_train_loader, data_val_loader, epoch, device):
        """
        Train the VAE for one epoch.

        Parameters
        ----------
        `optimizer` : torch.optim, the optimizer to use.
        `data_train_loader` : torch.utils.data.DataLoader, the data loader for the training data.
        `epoch` : int, the current epoch.
        `device` : torch.device, the device to use for training.
        """

        train_loss = 0
        for batch_idx, data in enumerate(data_train_loader):

            data = data.to(device)

            optimizer.zero_grad()

            y, z_mu, z_log_var = self.forward(data) 
            loss_vae = self.loss_function(data, y, z_mu, z_log_var) 
            loss_vae.backward()
            train_loss += loss_vae.item()
            optimizer.step() 

        avg_train_loss = train_loss / len(data_train_loader.dataset)   
        print('Epoch: {}\tAverage train loss: {:.4f}'.format(epoch, avg_train_loss))

        # Validation loss
        avg_val_loss = self.compute_test_loss(data_val_loader, device)
        print('\t\tAverage validation loss: {:.4f}'.format(avg_val_loss))

        return avg_train_loss, avg_val_loss
    
    def compute_test_loss(self, data_loader, device):
        """
        Compute the test (or validation) loss.

        Parameters:
        -----------
        `data_test_loader` : torch.utils.data.DataLoader, the data loader for the test data.
        `device` : torch.device, the device to use for testing.

        Returns:
        --------
        `avg_test_loss` : float, the average test loss.
        """
        
        loss = 0
        for batch_idx, data in enumerate(data_loader):

            data = data.to(device)

            y, z_mu, z_log_var = self.forward(data, test=True) 
            loss_vae = self.loss_function(data, y, z_mu, z_log_var)
            loss += loss_vae.item()

        avg_loss = loss / len(data_loader.dataset)
        return avg_loss

    def predict(self, x, device):
        """
        Run VAE in test mode: classify each pixel into one channel.

        Parameters
        ----------
        `x` : tensor, the input images.
        `device` : torch.device, the device to use for predicting.
        """
        x_hat, _, _ = self.forward( x.to(device), test=True )
        labels = torch.argmax(x_hat, dim=1)
        one_hot = torch.zeros_like(x_hat).scatter_(1, labels.unsqueeze(1), 1)
        return one_hot

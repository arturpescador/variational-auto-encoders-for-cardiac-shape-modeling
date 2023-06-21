import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import norm
import preprocessing as pre

def visualize(vae_model, input_tensor, device):
    """
    Visualize the input and the output of the VAE model.

    Parameters:
    -----------
    `vae_model` : VAE object, the trained VAE model
    `input` : tensor, the input images.
    `device` : torch.device, the device to use for predicting
    """
    input = input_tensor.cpu().detach().numpy()
    output_tensor = vae_model.predict( input_tensor, device )
    output = output_tensor.cpu().detach().numpy()
    
    for i in range(0, len(input)):
        fig, axs = plt.subplots( 1, 2, figsize=(8,4) )
        axs = axs.ravel()
        axs[0].imshow( np.moveaxis( input[i], [0,1,2], [2,0,1] )[:,:,1:] )
        axs[1].imshow( np.moveaxis( output[i], [0,1,2], [2,0,1] )[:,:,1:] )
        plt.show()

def plot_loss(train_loss, val_loss):
    """
    Plot the validation and test losses over epochs.
    
    Parameters:
    -----------
    `val_losses` : list, the validation losses over epochs.
    `test_losses` : list, the test losses over epochs.
    """
    epochs = len(train_loss)
    print("Total epochs: ", epochs)

    plt.plot(range(epochs), train_loss, label='train')
    plt.plot(range(epochs), val_loss, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

def generate_latent(model, dataloader, device):
    """
    Generate latent space vectors for all the images in the dataset

    Parameters:
    -----------
    model: VAE model
    dataloader: PyTorch dataloader
    device: torch.device

    Returns:
    --------
    mus: torch.Tensor containing the mu vectors for each image
    logvars: torch.Tensor containing the logvar vectors for each image       
    """
    mus = []
    logvars = []
    
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # No need to track gradients
        for data in dataloader:
            mu, logvar = model.encoder(data.to(device))  # Get mu and logvar
            mus.append(mu)
            logvars.append(logvar)
    
    mus = torch.cat(mus, dim=0)  # Concatenate all mu and logvar tensors
    logvars = torch.cat(logvars, dim=0)
    
    return mus, logvars
    
def check_distribution(mus, logvars):
    """
    Check that the distribution of the latent space vectors is close to a standard normal distribution

    Parameters:
    -----------
    mus: torch.Tensor containing the mu vectors for each image
    logvars: torch.Tensor containing the logvar vectors for each image
    """

    mus = mus.cpu().numpy()
    stds = np.sqrt(np.exp2(logvars.cpu().numpy()))  # Calculate standard deviations from log variances

    mu_mean = np.mean(mus)
    mu_std = np.std(mus)

    std_mean = np.mean(stds)
    std_std = np.std(stds)

    print(f"Mu: mean={mu_mean}, std={mu_std}") # Check that the mean and std are close to 0 and 1 respectively
    print(f"Std: mean={std_mean}, std={std_std}") # Check that the mean and std are close to 0 and 1 respectively
    


def visualize_generated_images(generated_samples):
    """
    Visualize the generated samples.

    Parameters:
    -----------
    `generated_samples` : numpy array, the generated samples
    """
    num_samples = generated_samples.shape[0]
    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

    for i in range(num_samples):
        axs[i].imshow(np.moveaxis(generated_samples[i], [0, 1, 2], [2, 0, 1]))
        axs[i].axis('off')

    plt.show()

import torch
import numpy as np

def sorted_recon_losses(model, test_loader, device):
    """
    Computes the reconstruction loss for each image in the test set and sorts them in ascending order.
    
    Parameters
    ----------
    `model` : VAE - trained model
    `test_loader` : DataLoader - test set
    `device` : str - device on which to run the computations

    Returns
    -------
    `recon_losses` : np.array - sorted reconstruction losses
    `original_images` : np.array - original images
    `reconstructed_images` : np.array - reconstructed images
    """
    
    # List to store reconstruction losses, original images and their reconstructions
    recon_losses = []
    original_images = []
    reconstructed_images = []

    # Model in evaluation mode
    model.eval()

    # Loop over all batches in the test set
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to the device
            batch = batch.to(device)

            # Forward pass and reconstruct the input
            recon, _, _ = model(batch)

            # Compute reconstruction loss (MSE)
            loss = model.soft_dice_loss(recon, batch, reduction ="none").detach().cpu()

            # Append loss, original images and their reconstructions to lists
            recon_losses.append(loss)
            original_images.append(batch.detach().cpu())
            reconstructed_images.append(recon.detach().cpu())

    # Concatenate all batches
    recon_losses = np.concatenate(recon_losses)
    original_images = torch.cat(original_images)
    reconstructed_images = torch.cat(reconstructed_images)

    # Sort by reconstruction loss
    indices = np.argsort(recon_losses)
    
    return recon_losses, original_images, reconstructed_images, indices

def visualize_generated_images(generated_images):
    """
    Visualize the generated images.

    Parameters:
    -----------
    `generated_images` : tensor, the generated samples.
    """
    generated_images = generated_images.cpu().detach().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(20,4))
    for i in range(0, len(generated_images)):
        axs[i].imshow(np.moveaxis(generated_images[i], [0, 1, 2], [2, 0, 1])[:, :, 1:])
        axs[i].axis('off')  # Turn off the axis labels

def visualize_generated_images1(generated_samples):
    """
    Visualize the generated samples.

    Parameters:
    -----------
    `generated_samples` : tensor, the generated samples.
    """
    samples = generated_samples.cpu().detach().numpy()
    

    fig, axs = plt.subplots(1, len(samples), figsize=(len(samples) * 2, 2))

    for i in range(len(samples)):
        axs[i].imshow(samples[i])

    
    plt.show()


def visualize_generated_images2(imgs):
    """
    Visualize the generated samples.

    Parameters:
    -----------
    `generated_samples` : tensor, the generated samples.
    """
    images = imgs.cpu().detach().numpy()
    r = 1
    c = images.shape[0]
    fig, axs = plt.subplots(r, c)
    for j in range(c):
      #black and white images
      axs[j].imshow(images[j, :,:],)
      axs[j].axis('off')
    plt.show()
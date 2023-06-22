import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import norm

import model as m
import preprocessing as pre

def visualize(vae_model, input_tensor, device, save=False, path=None):
    """
    Visualize the input and the output of the VAE model, runs the model on the input tensor.

    Parameters:
    -----------
    `vae_model` : VAE object, the trained VAE model
    `input` : tensor, the input images.
    `device` : torch.device, the device to use for predicting
    `save` : bool, whether to save the images or not
    `path` : str, the path to save the images
    """
    input = input_tensor.cpu().detach().numpy()
    output_tensor = vae_model.predict( input_tensor, device )
    output = output_tensor.cpu().detach().numpy()
    
    fig, axs = plt.subplots( 2, len(input), figsize=(len(input)*2, 4) )

    for i in range(0, len(input)):
        axs[0, i].imshow( np.moveaxis( input[i], [0,1,2], [2,0,1] )[:,:,1:] )
        axs[0, i].axis('off')
        axs[1, i].imshow( np.moveaxis( output[i], [0,1,2], [2,0,1] )[:,:,1:] )
        axs[1, i].axis('off')
    
    if save:
        plt.savefig(path+'images/input_output_lamb{}.png'.format(vae_model.lamb), dpi=300, bbox_inches='tight')
    plt.show()

def visualize_generated_images(generated_samples_tensor, save=False, path=None, lamb=None):
    """
    Visualize the generated samples.

    Parameters:
    -----------
    `generated_samples_tensor` : tensor, the generated samples.
    `save` : bool, whether to save the images or not
    `path` : str, the path to save the images
    `lamb` : float, the lambda value
    """
    generated_samples = generated_samples_tensor.cpu().detach().numpy()
    num_samples = generated_samples.shape[0]
    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

    for i in range(num_samples):
        axs[i].imshow(np.moveaxis(generated_samples[i], [0, 1, 2], [2, 0, 1])[:,:,1:])
        axs[i].axis('off')

    if save:
        plt.savefig(path+'images/generated_lamb{}.png'.format(lamb), dpi=300, bbox_inches='tight')
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

def save_loss(train_loss, val_loss, test_loss, path, filename_sufix):
    """
    Save the train and validation losses over epochs and test loss to a file.
    
    Parameters:
    -----------
    `val_losses` : list, the validation losses over epochs.
    `test_losses` : list, the final test loss.
    `path` : str, the path to save the file.
    `filename_sufix` : str, the filename sufix.
    """
    np.savez( path+'loss_'+filename_sufix, np.array(train_loss+val_loss+[test_loss]) )

def load_loss(path, filename_sufix):
    """
    Load the train and validation losses over epochs and test loss from a file.
    
    Parameters:
    -----------
    `path` : str, the path to load the file.
    `filename_sufix` : str, the filename sufix.

    Returns:
    --------
    `train_loss` : list, the train losses over epochs.
    `val_loss` : list, the validation losses over epochs.
    `test_loss` : float, the final test loss.
    """
    losses = np.load( path+'loss_'+filename_sufix+'.npz' )['arr_0'].tolist()
    test_loss = losses.pop()
    train_loss = losses[:len(losses)//2]
    val_loss = losses[len(losses)//2:]
    return  train_loss, val_loss, test_loss

def evaluate_lambda(train_loader, val_loader, test_loader, lambda_list, device, path):
    """
    Evaluate the VAE model for different values of lambda.
    
    Parameters:
    -----------
    `train_loader` : PyTorch dataloader, the training set.
    `val_loader` : PyTorch dataloader, the validation set.
    `test_loader` : PyTorch dataloader, the test set.
    `lambda_list` : list, the list of lambda values to evaluate.
    `device` : torch.device, the device to use for training.
    `path` : str, the path to save the files.
    """

    # Define the dimensions of the input space
    n_channels = 4
    n_rows = n_cols = next(iter(test_loader))[0].shape[-1]
    z_dim = 32

    # Define the optimizer parameters
    num_epochs = 100
    learning_rate = 6e-5
    l2 = 0.01

    for lamb in lambda_list:

        print("\rLambda:{}".format(lamb))

        # Create an instance of the VAE model
        model = m.VAE(n_rows, n_cols, n_channels, z_dim, lamb).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

        train_loss = []
        val_loss = []
        for epoch in range(num_epochs):
            print("\r{}/{}".format(epoch, num_epochs), end='')
            new_train_loss, new_val_loss = model.train_one_epoch(optimizer, train_loader, val_loader, epoch=epoch, device=device, verbose=False)
            train_loss.append(new_train_loss)
            val_loss.append(new_val_loss)

        test_loss = model.compute_test_loss(test_loader, device)

        # Save losses
        save_loss(train_loss, val_loss, test_loss, path, str(lamb))

        # Save model
        torch.save(model.state_dict(), path+'./model_{}.pt'.format(lamb))

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
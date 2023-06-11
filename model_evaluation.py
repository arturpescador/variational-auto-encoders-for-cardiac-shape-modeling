import torch 
import numpy as np
import matplotlib.pyplot as plt
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

def soft_dice_loss(output, target, smooth=1e-5):
    """
    Compute the soft Dice loss between the predicted output and the target.

    Args:
        output (torch.Tensor): Predicted output from the VAE model.
        target (torch.Tensor): Ground truth data.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Soft Dice loss.
    """
    # Flatten the predicted output and the target
    output = output.view(-1)
    target = target.view(-1)

    intersection = (output * target).sum()
    dice = (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
    dice_loss = 1.0 - dice

    return dice_loss

def calculate_average_dice_loss(model, test_loader, device):
    """
    Calculate the average soft Dice loss for the test data.

    Parameters:
    -----------
    `model` : VAE object, the trained VAE model
    `test_loader` : torch.utils.data.DataLoader, the data loader for the test data.
    `device` : torch.device, the device to use for predicting

    Returns:
    --------
    `avg_dice_loss` : float, the average soft Dice loss for the test data.
    """

    # Initialize list to store soft Dice losses
    dice_losses = []

    # Iterate over the test data loader
    for data in test_loader:
        # Move the data to the device
        data = data.to(device)

        # Forward pass through the model to obtain the reconstructed output
        with torch.no_grad():
            output, _, _ = model(data)

        # Calculate the soft Dice loss
        dice_loss = soft_dice_loss(output, data)
        dice_losses.append(dice_loss.item())

    # Calculate the average soft Dice loss
    avg_dice_loss = np.mean(dice_losses)

    return avg_dice_loss
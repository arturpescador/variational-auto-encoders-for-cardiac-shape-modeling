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
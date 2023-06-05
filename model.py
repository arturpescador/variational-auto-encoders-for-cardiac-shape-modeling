import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn 

def soft_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Calculate the soft Dice loss between the ground truth and predicted masks.

    Parameters
    ----------
    `y_true` : tensor, the ground truth mask.
    
    `y_pred` : tensor, the predicted mask.

    `smooth` : float, to avoid division by zero. Default is `1e-6`.
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])

    dice_coef = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - tf.reduce_mean(dice_coef)

    return dice_loss


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
    optimizer = torch.optim.Adam(classification_model.parameters(), lr=learning_rate)
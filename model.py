import tensorflow as tf
import numpy as np

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
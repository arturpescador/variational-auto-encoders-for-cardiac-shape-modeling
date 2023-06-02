import os
import nibabel as nib
import matplotlib.pyplot as plt
import re
import numpy as np
import imutils
from skimage.transform import resize

def nii_reader(path):
    """
    Read a nii file and return the image as a numpy array
    """
    nii = nib.load(path)
    return nii.get_fdata()

def visualize_image_mask(image, mask, depth_size):
    """
    Visualize an image and its mask
    """

    # Visualize the slices of the image
    fig, ax = plt.subplots(1, depth_size, figsize=(25, 8))
    for j in range(depth_size):
        ax[j].imshow(image[:,:,j], cmap='gray')  # show one single slice of each image
        ax[j].axis('off')
        ax[j].set_title('Slice: {}'.format(j))
    plt.show()

    # Visualize the slices of the ground truth image
    fig, ax = plt.subplots(1, depth_size, figsize=(25, 8))
    for j in range(depth_size):
        ax[j].imshow(mask[:,:,j], cmap='gray')  # show one single slice of each image
        ax[j].axis('off')
        ax[j].set_title('Slice: {}'.format(j))
    plt.show()

def visualize_mask(mask, show_axis=False):
    """
    Visualize a 3D segmentation mask
    """

    # Visualize the slices of the ground truth image
    fig, ax = plt.subplots(1, mask.shape[-1], figsize=(25, 8))
    for j in range(mask.shape[-1]):
        ax[j].imshow(mask[:,:,j], cmap='gray')  # show one single slice of each image
        ax[j].axis(show_axis)
        ax[j].set_title('Slice: {}'.format(j))
    plt.show()

def visualize_2d_mask(mask):
    """
    Visualize a 2D segmentation mask
    """

    plt.figure( figsize=(4,4) )
    plt.imshow( mask, cmap='gray' )
    plt.axis(False)
    plt.show()

def visualize_multichannel_mask(mask):
    """
    Visualize a 3-channel segmentation mask, each channel corresponds to a different heart structure
    """

    plt.figure( figsize=(4,4) )
    plt.imshow( mask )
    plt.show()

def preprocess_files_acdc(folder, nb_files, test=False):
    """
    Load the images and masks from the ACDC dataset
    """
    images_ED = []
    images_ES = []
    masks_ED = []
    masks_ES = []

    if test:
        start = 101
    else:
        start = 1

    for i in range(start, nb_files + start):
        files_folder = []
        frame_number = []
        patient_folder = os.path.join(folder, 'patient' + str(i).zfill(3))

        for file in os.listdir(patient_folder):
            if file.endswith('.nii.gz'):
                files_folder.append(file)
                if 'gt' in file and 'frame' in file:
                    match = re.search(r'frame(\d+)', file)
                    frame_number.append(int(match.group(1)))

        ed_frame = min(frame_number)

        for file in files_folder:
            if 'gt' in file:
                if 'frame' in file:
                    match = re.search(r'frame(\d+)', file)
                    frame_number = int(match.group(1))
                    if frame_number == ed_frame:  # ED  = end diastolic
                        masks_ED.append(os.path.join(patient_folder, file))
                    else:  # ES = end systolic
                        masks_ES.append(os.path.join(patient_folder, file))
            else:
                if 'frame' in file:
                    match = re.search(r'frame(\d+)', file)
                    frame_number = int(match.group(1))
                    if frame_number == ed_frame:  # ED  = end diastolic
                        images_ED.append(os.path.join(patient_folder, file))
                    else:  # ES = end systolic
                        images_ES.append(os.path.join(patient_folder, file))

    return images_ED, masks_ED, images_ES, masks_ES

def heart_mask_loader(masks_patients):
    """
    Load the masks of the heart from the ACDC dataset

    Parameters:
    -----------
    `masks_patients`: list of paths to heart masks

    Returns:
    --------
    `masks`: list of heart masks
    """

    masks = [ nii_reader(path) for path in masks_patients ]

    return masks

def align_heart_mask( masks ):
    """
    Rotates the heart masks so that the relative position of the LV and RV is always the same

    Parameters:
    -----------
    `masks`: list of heart masks

    Returns:
    --------
    `rotated_masks`: list of rotated heart masks
    """

    rotated_masks = []

    for mask in masks:
        rv_location = np.argwhere(mask == 1)
        rv_center = np.sum( rv_location, axis=0 )/rv_location.shape[0]

        lv_location = np.argwhere(mask == 3)
        lv_center = np.sum( lv_location, axis=0 )/lv_location.shape[0]

        rad = np.arctan2(rv_center[0]-lv_center[0], rv_center[1]-lv_center[1])

        rotated_masks.append( imutils.rotate( mask, rad*180/np.pi ) )

    return rotated_masks

def crop_heart_mask( masks ):
    """
    Crops excedent background from the heart masks

    Parameters:
    -----------
    `masks`: list of heart masks

    Returns:
    --------
    `cropped_masks`: list of (square) cropped heart masks
    """   

    cropped_masks = []
    for mask in masks:
        not_background = np.argwhere(mask != 0)
        min_row = np.min(not_background[:, 0])
        max_row = np.max(not_background[:, 0])
        min_col = np.min(not_background[:, 1])
        max_col = np.max(not_background[:, 1])
        s = max(max_row - min_row, max_col - min_col)
        cropped_mask = mask[min_row:min_row+s, min_col:min_col+s]
        cropped_masks.append( cropped_mask )
    
    return cropped_masks

def resize_heart_mask(masks, s=256):
    """
    Resamples image (nearest neighbour sampling) to the desired size

    Parameters:
    -----------
    `masks`: list of heart masks
    `s`: size of the output square images

    Returns:
    --------
    `resized_masks`: list of resized heart masks
    """ 
    resized_masks = [ resize( mask, (s,s,mask.shape[-1]), order=0, preserve_range=0 ) for mask in masks ]

    return resized_masks

def convert_3D_to_2D(masks):
    """
    Disassemble frames as independent 2D images

    Parameters:
    -----------
    `masks`: list of 3D heart masks

    Returns:
    --------
    `masks_2D`: list of 2D heart masks
    """ 
    concat = np.concatenate( masks, axis=-1 ) # concatenate all the masks in a single 3D array
    masks_2D = [ concat[:,:,i] for i in range(concat.shape[2]) ]
    return masks_2D


def heart_mask_extraction_v1(masks_patients):
    """
    Extract each cavity of the heart and the myocardium.
    Eliminates other elements that are not the desired ones.

    Parameters:
    -----------
    `images_seg`: segmented images

    Returns:
    --------
    `masks_rv`: mask for the right ventricle
    `masks_myo`: mask for the myocardium
    `masks_lv`: mask for the left ventricle
    """

    nb_patients = len(masks_patients)
    masks_rv = []
    masks_myo = []
    masks_lv = []

    # Iterate through the patients to get their masks
    for i in range(nb_patients):
        masks = nii_reader(masks_patients[i])
        seg_rv = []
        seg_myo = []
        seg_lv = []

        # Take the mask for each slice of the patient
        for mask in masks:
            seg_rv.append(np.where(mask == 1, 1, 0))
            seg_myo.append(np.where(mask == 2, 1, 0))
            seg_lv.append(np.where(mask == 3, 1, 0))

        masks_rv.append(seg_rv)    
        masks_myo.append(seg_myo)
        masks_lv.append(seg_lv)

    return masks_rv, masks_myo, masks_lv

def heart_mask_extraction(masks):
    """
    Each structure will be mapped to a different binary channel.

    Parameters:
    -----------
    `masks`: list of single channel 2d masks

    Returns:
    --------
    `mew_masks`: list of 4-channel binary masks
    """

    new_masks = []
    for mask in masks:
        corrected_mask = np.round(mask)
        new_mask = np.zeros((masks[0].shape[0], masks[0].shape[1], 4))
        new_mask[:,:,0] = np.where(corrected_mask == 0, 1, 0)
        new_mask[:,:,1] = np.where(corrected_mask == 1, 1, 0)
        new_mask[:,:,2] = np.where(corrected_mask == 2, 1, 0)
        new_mask[:,:,3] = np.where(corrected_mask == 3, 1, 0)
        new_masks.append(new_mask)

    return new_masks
import os
import nibabel as nib
import matplotlib.pyplot as plt
import re
import numpy as np
import imutils

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

def visualize_mask(mask):
    """
    Visualize a segmentation mask
    """

    # Visualize the slices of the ground truth image
    fig, ax = plt.subplots(1, mask.shape[-1], figsize=(25, 8))
    for j in range(mask.shape[-1]):
        ax[j].imshow(mask[:,:,j], cmap='gray')  # show one single slice of each image
        ax[j].axis('off')
        ax[j].set_title('Slice: {}'.format(j))
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

def heart_mask_extraction(masks_patients):
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
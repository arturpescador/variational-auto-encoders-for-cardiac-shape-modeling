import os
import nibabel as nib
import matplotlib.pyplot as plt
import re

def nii_reader(path):
    """
    Read a nii file and return the image as a numpy array
    """
    nii = nib.load(path)
    return nii.get_fdata()

def visualize_image_mask(image, mask):
    """
    Visualize an image and its mask
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(mask, cmap='gray')
    plt.show()


def preprocess_files_acdc(folder, nb_files):
    """
    Load the images and masks from the ACDC dataset
    """
    images_ED = []
    images_ES = []
    masks_ED = []
    masks_ES = []

    for i in range(1, nb_files + 1):
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
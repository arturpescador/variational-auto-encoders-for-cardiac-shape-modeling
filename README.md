# Variational Auto Encoders for cardiac shape modeling

This project aims to implement a convolutional Variational Autoencoder (VAE) to create a mapping between a latent space of cardiac shape and 2D segmentation masks of the left ventricle, right ventricle, and myocardium. The VAE takes 4-channel 2D images as input, where each image represents the segmentation mask of a cardiac MRI slice, and the channels correspond to the segmentation masks of the background, left ventricle, right ventricle, and myocardium. The objective is to evaluate the quality of the reconstruction of the segmentation masks.

## References

The main articles consulted during the development of this project are:

1. Biffi, C., et al. (2018) Learning interpretable anatomical features through deep generative models: Application to cardiac remodeling. International conference on medical image computing and computer-assisted intervention, 464-471.

2. Painchaud, N., et al. (2020) Cardiac Segmentation with Strong Anatomical Guarantees. IEEE Transactions on Medical Imaging 39(11), 3703-3713.

## Files

The project consists of the following files:

- `preprocessing.py`: implementation of the methods required to load and pre-process the dataset.

- `model.py`: implementation of the VAE model.

- `model_evaluation.py`: implementation of the methods required to evaluate the model.

- `report.ipynb`: jupyter notebook containing the code and explanations for the project.

## Contributors

This project was developed by the following group members as part of the Computational Photography / Patch Methods course (IMA206) at Télécom-Paris for the academic year 2022/2023, under the supervision of Loïc Le Folgoc and Elsa Angelini:

- Alice Valença De Lorenci
- Artur Dandolini Pescador
- Giulia Mannaioli
- Lais Isabelle Alves dos Santos

## Dataset

The dataset used for this project is sourced from the [ACDC Challenge](https://acdc.creatis.insa-lyon.fr/). It consists of a training-validation set with 100 subjects and a test set with 50 subjects. The dataset provides cardiac MRI images and their corresponding segmentation maps for end systole (ES) and end diastole (ED) phases. The segmentation map includes the following structures (with respective labels):

0. Background
1. Right ventricle cavity (RV)
2. Myocardium (MY)
3. Left ventricle cavity (LV)







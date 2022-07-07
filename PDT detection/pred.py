# -*- coding: utf-8 -*-
import cv2
import keras
from matplotlib import pyplot as plt

from Utils.utils import show
from Models.Model import dice_loss, combined, dice_coeff
from Utils.segmentation import segment_potatoes

import time

model_mask = keras.models.load_model('Trained Models/mask_yellow.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
model_contour = keras.models.load_model('Trained Models/cnt_yellow.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
model_target = keras.models.load_model('Trained Models/targets4.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

for no in range(7, 8):
    path = f'PDT detection/SolanumTuberosum/Test_images/test{no}.jpg'
    color_img, d, h, mask, pred_mask = segment_potatoes(path, model_mask, model_contour, model_target, patch_size=256, resize=(2048, 1536), norm_fact=255)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(ncols=4, figsize=(10, 4))
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[1].imshow(mask, cmap='gray')
    ax[2].imshow(pred_mask, cmap='gray')
    ax[3].imshow(color_img)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')

    ax[0].set_title('Image originale', fontsize=8)
    ax[1].set_title('Masque prédit brute', fontsize=8)
    ax[2].set_title('Contours soustraits + post-processing', fontsize=8)
    ax[3].set_title('Prédiction finale', fontsize=8)

    plt.tight_layout()
    fig.savefig(f'Results/test{no}.png', dpi=500)

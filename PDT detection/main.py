# -*- coding: utf-8 -*-
import cv2
import keras
from matplotlib import pyplot as plt

from Utils.utils import show
from Models.Model import dice_loss, combined, dice_coeff, weighted_bce
from Utils.segmentation import segment_potatoes

import time

model_contour = keras.models.load_model('Trained Models/cnt_wbce_8c_bs4_100ep_BNFalse_reg0.0005.h5', custom_objects={'weighted_bce': weighted_bce, 'dice_coeff': dice_coeff})
model_target = keras.models.load_model('Trained Models/targets4.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

models_mask = ['mask_blue_gray_bg', 'mask_8c_bs4_100ep_BNFalse_reg0.0005']
norm_factors = [1, 255]
name = 'test11'
path = f'PDT detection/SolanumTuberosum/Test_images/{name}.jpg'
f = 255

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax = axes.ravel()

for i, (mm, f) in enumerate(zip(models_mask, norm_factors)):
    model_mask = keras.models.load_model(f'Trained Models/{mm}.h5', custom_objects={'combined': combined, 'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
    color_img, d, h, targets, mask = segment_potatoes(path, model_mask, model_contour, model_target, patch_size=256, resize=(2048, 1536), norm_fact=f)
    ax[i].imshow(mask)
    ax[i].axis('off')
    ax[i].set_title(mm)

plt.show()

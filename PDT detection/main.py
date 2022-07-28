# -*- coding: utf-8 -*-
import cv2
import keras
from matplotlib import pyplot as plt

from Utils.utils import show
from Models.Model import dice_loss, combined, dice_coeff
from Utils.segmentation import segment_potatoes

import time

model_contour = keras.models.load_model('Trained Models/cnt_yellow.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
model_target = keras.models.load_model('Trained Models/targets.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

models_mask = ['mask_blue_gray_bg', 'mask_colors', 'mask_+5pics', 'mask_colors++']
norm_factors = [1, 255, 255, 255]
name = 'test15'
path = f'PDT detection/SolanumTuberosum/Test_images/test11.jpg'
f = 255

model_mask = keras.models.load_model(f'Trained Models/mask_yellow.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
color_img, d, h, mask, mask2 = segment_potatoes(path, model_mask, model_contour, model_target, patch_size=256, resize=(2048, 1536), norm_fact=f)
show(color_img)
# fig.savefig(f'Results/{name}.png', dpi=500)
# show(color_img, dims=(1000, 750))
# cv2.imwrite('gallerie_low_res.png', color_img)

# heights = [pixels_to_mm(h[i], 72, 1.97) for i in range(len(h))]
# diameters = [pixels_to_mm(d[i], 72, 1.97) for i in range(len(d))]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), sharey='all')
ax1.hist(h, bins=[50, 70, 90, 110, 130, 150, 170, 190], alpha=0.5, histtype='bar', ec='black')
ax1.set_xlabel('Largeur [mm]')
ax2.set_xlabel('Longueur [mm]')
ax1.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax2.hist(d, bins=[20, 30, 40, 50, 60, 70, 80, 90, 100], alpha=0.5, histtype='bar', ec='black')
plt.show()


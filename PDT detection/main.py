# -*- coding: utf-8 -*-
import cv2
import keras
from matplotlib import pyplot as plt

from Utils.utils import show
from Models.Model import dice_loss, combined, dice_coeff
from Utils.segmentation import segment_potatoes

import time

model_contour = keras.models.load_model('Trained Models/contours_final.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
model_target = keras.models.load_model('Trained Models/targets4.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

models_mask = ['mask_blue_gray_bg', 'mask_colors', 'mask_+5pics', 'mask_colors++']
norm_factors = [1, 255, 255, 255]
name = 'test15'
path = f'PDT detection/SolanumTuberosum/Test_images/{name}.jpg'

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ax = axes.ravel()

for i, (model_name, f) in enumerate(zip(models_mask, norm_factors)):
    model_mask = keras.models.load_model(f'Trained Models/{model_name}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
    color_img, d, h, mask = segment_potatoes(path, model_mask, model_contour, model_target, patch_size=256, resize=(2048, 1536), norm_fact=f)
    ax[i].imshow(mask)
    ax[i].axis('off')
    ax[i].set_title(model_name)
plt.tight_layout()
fig.savefig(f'Results/{name}.png', dpi=500)
# show(color_img, dims=(1000, 750))
# cv2.imwrite('gallerie_low_res.png', color_img)

# heights = [pixels_to_mm(h[i], 72, 1.97) for i in range(len(h))]
# diameters = [pixels_to_mm(d[i], 72, 1.97) for i in range(len(d))]

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), sharey='all')
# ax1.hist(heights, bins=[50, 70, 90, 110, 130, 150, 170, 190], alpha=0.5, histtype='bar', ec='black')
# ax1.set_xlabel('Largeur [mm]')
# ax2.set_xlabel('Longueur [mm]')
# ax1.set_yticks([0, 5, 10, 15, 20, 25, 30])
# ax2.hist(diameters, bins=[20, 30, 40, 50, 60, 70, 80, 90, 100], alpha=0.5, histtype='bar', ec='black')
# plt.show()
#
# calibres = {'3po+': 0, '3po': 0, '2 1/2 po': 0, '2 1/4 po': 0, '2po': 0, '1 7/8 po': 0, '1 3/4 po': 0}
# for mm in d:
#     po = mm/25.4
#     if po >= 3:
#         calibres['3po+'] += 1
#     elif po >= 2.5:
#         calibres['3po'] += 1
#     elif po >= 2.25:
#         calibres['2 1/2 po'] += 1
#     elif po >= 2:
#         calibres['2 1/4 po'] += 1
#     elif po >= 1.875:
#         calibres['2po'] += 1
#     elif po >= 1.75:
#         calibres['1 7/8 po'] += 1
#     else:
#         calibres['1 3/4 po'] += 1
#
# for cal, num in calibres.items():
#     print(f'{cal}: {num}')

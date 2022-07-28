# -*- coding: utf-8 -*-
import base64
import json

import cv2
from flask import jsonify
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

from Utils.utils import show, show_QQplot_width_length, get_calibres
from Models.Model import dice_loss, combined, dice_coeff
from Utils.segmentation import segment_potatoes

import time

# Load the necessary models
model_mask = keras.models.load_model('Trained Models/mask_yellow.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
model_contour = keras.models.load_model('Trained Models/cnt_yellow.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
model_target = keras.models.load_model('Trained Models/targets.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

# Get full predictions
filename = 'test11'
variety = 'mountain_gem'
path = f'PDT detection/SolanumTuberosum/Test_images/{filename}.jpg'
color_img, d, h, targets = segment_potatoes(path, model_mask, model_contour, model_target, patch_size=256, resize=(2048, 1536), norm_fact=255, dev_mode=True)
# Trace QQ plot to compare real and predicted distribution
real_h = np.loadtxt(f'PDT detection/SolanumTuberosum/Dimensions/{variety}.txt', usecols=0, skiprows=2)
# real_d = np.loadtxt(f'PDT detection/SolanumTuberosum/Dimensions/{variety}.txt', usecols=1, skiprows=2)
# show_QQplot_width_length(d=d, h=h, real_d=real_d, real_h=real_h, variety=variety)

# Print other informations regarding the sample
# if variety == 'burbank':
#     d_norm = [(i - 39)/(84 - 39) for i in d]
#     h_norm = [(i - 62)/(179 - 62) for i in h]
# else:
#     d_norm = [(i - 39) / (81 - 39) for i in d]
#     h_norm = [(i - 48) / (190 - 48) for i in h]

# model = keras.models.load_model(f"Trained models/{variety}_weight.h5")
# sample = np.stack([h_norm, d_norm], axis=-1)
# weights = model.predict(sample)
# sum_weight = np.sum(weights)
# calibres = get_calibres(d)
# print(f'Total weight: {sum_weight:.1f} g')
# print(f'Average length: {np.mean(h):.1f} mm')
# print(f'Average width: {np.mean(d):.1f} mm')
# print('Calibres:')
# for cal, num in calibres.items():
#     print(f'{cal}: {num}')


# Display original image with predicted identifications
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
# img = cv2.imread(path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(ncols=1, figsize=(10, 5))
plt.imshow(color_img)
plt.axis('off')

# ax[0].set_title('Image originale', fontsize=8)
plt.title('Prédiction finale', fontsize=8)

plt.tight_layout()
plt.show()
# fig.savefig(f'Results/{filename}.png', dpi=500)

import keras
import numpy as np
import cv2
import os
from PIL import Image
from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt
from skimage.filters.rank import modal
from skimage.morphology import rectangle

from segmentation import full_prediction
from Model import dice_coeff, dice_loss, combined

model_mask = keras.models.load_model('Trained Models/mask_final.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
model_contour = keras.models.load_model('Trained Models/contours_final.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})

path = f'SolanumTuberosum/Test_images/test7.jpg'

pred_mask = full_prediction(model_mask, path, 256)
pred_mask = modal(pred_mask, rectangle(3, 3))

pred_contour = full_prediction(model_contour, path, 256)
pred_contour = modal(pred_contour, rectangle(3, 3))

pred = pred_mask - pred_contour
pred[pred < 5] = 0

fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
ax = ax.ravel()
ax[0].imshow(load_img(path, color_mode='rgb'), cmap='gray')
ax[1].imshow(pred_mask, cmap='gray')
ax[2].imshow(pred_contour, cmap='gray')
ax[3].imshow(pred, cmap='gray')

ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[3].axis('off')

# ax[0].set_title('Image originale complète')
# ax[1].set_title('Masque binaire complet prédit')
# ax[2].set_title('Contours complets prédits')
# ax[3].set_title('Masque - contours')

plt.tight_layout()
plt.show()
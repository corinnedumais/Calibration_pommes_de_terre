import keras
import numpy as np
import cv2
from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects

from Utils.gapfilling import fill_gaps
from segmentation import full_prediction, segment_potatoes
from Models.Model import dice_coeff, dice_loss, combined

mask_model = keras.models.load_model('Trained Models/mask_blue_gray_bg.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
contours_model = keras.models.load_model('Trained Models/contours_final.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})

path = f'PDT detection/SolanumTuberosum/Test_images/test1.jpg'

# Mask and contour predictions
f = 1
pred_mask = full_prediction(mask_model, path, 256, resize=(2048, 1536), norm_fact=f)
pred_contour = full_prediction(contours_model, path, 256, resize=(2048, 1536), norm_fact=f)

# Modal filter to eliminate artifacts at the junction of the predicted tiles
pred_mask = modal(pred_mask, rectangle(5, 5))
mask = pred_mask.copy()
pred_contour = modal(pred_contour, rectangle(5, 5))

pred_contour = cv2.ximgproc.thinning(pred_contour) / 255
pred_contour = pred_contour.astype(np.uint8)

pred_contour = remove_small_objects(label(pred_contour), 72)
pred_contour[pred_contour != 0] = 1

###############################

pred_contour_gp = fill_gaps(pred_contour, n_iterations=10)
pred_mask_gp = pred_mask.copy()

###############################################

# prediction without gap filling
pred_contour[pred_contour != 0] = 1
pred_contour = pred_contour.astype(np.float32)
pred_contour = cv2.dilate(pred_contour, np.ones((3, 3)))
pred_contour = pred_contour.astype(np.uint8)

pred_mask[pred_contour == 1] = 0

pred_mask = remove_small_objects(label(pred_mask), 4000)
pred_mask[pred_mask != 0] = 255
pred_mask = pred_mask.astype(np.uint8)

# prediction with gap filling
pred_contour_gp[pred_contour_gp != 0] = 1
pred_contour_gp = pred_contour_gp.astype(np.float32)
pred_contour_gp = cv2.dilate(pred_contour_gp, np.ones((3, 3)))
pred_contour_gp = pred_contour_gp.astype(np.uint8)

pred_mask_gp[pred_contour_gp == 1] = 0

pred_mask_gp = remove_small_objects(label(pred_mask_gp), 4000)
pred_mask_gp[pred_mask_gp != 0] = 255
pred_mask_gp = pred_mask_gp.astype(np.uint8)

fig, ax = plt.subplots(ncols=5, figsize=(18, 4))
ax = ax.ravel()
ax[0].imshow(load_img(path, color_mode='rgb'), cmap='gray')
ax[1].imshow(mask, cmap='gray')
ax[2].imshow(pred_mask, cmap='gray')
ax[3].imshow(pred_mask_gp, cmap='gray')
color_img, d, h, m = segment_potatoes(path, mask_model, contours_model, None, patch_size=256, resize=(2048, 1536), norm_fact=f)
ax[4].imshow(color_img)

ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[3].axis('off')
ax[4].axis('off')

ax[0].set_title('Image originale complète')
ax[1].set_title('Masque binaire complet prédit')
ax[2].set_title('Masque - contours')
ax[3].set_title('Gap filling')
ax[4].set_title('Ellipses')

plt.tight_layout()
plt.show()

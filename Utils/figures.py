import keras
import numpy as np
import cv2
from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects

from segmentation import full_prediction
from Models.Model import dice_coeff, dice_loss, combined

mask_model = keras.models.load_model('Trained Models/mask_final.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
contours_model = keras.models.load_model('Trained Models/contours_final.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})

path = f'SolanumTuberosum/Test patates/gallerie.jpg'

# Mask and contour predictions
pred_mask = full_prediction(mask_model, path, 256, resize=(2048, 1536))
pred_contour = full_prediction(contours_model, path, 256, resize=(2048, 1536))

# Modal filter to eliminate artifacts at the junction of the predicted tiles
pred_mask = modal(pred_mask, rectangle(5, 5))
pred_contour = modal(pred_contour, rectangle(5, 5))
pred_contour = cv2.erode(pred_contour, np.ones((3, 3)), iterations=1)

pred_contour = cv2.ximgproc.thinning(pred_contour) / 255
pred_contour = pred_contour.astype(np.uint8)

pred_contour = remove_small_objects(label(pred_contour), 72)
pred_contour[pred_contour != 0] = 1

###############################

# start_time = time.time()
# pred_contour = fill_gaps(pred_contour, n_iterations=10)
# print(f'Execution time: {time.time() - start_time: .3f} seconds')

###############################################

pred_contour[pred_contour != 0] = 1
pred_contour = pred_contour.astype(np.float32)
pred_contour = cv2.dilate(pred_contour, np.ones((3, 3)))
pred_contour = pred_contour.astype(np.uint8)

# Subtraction of the two masks and elimination of negative values
pred_mask[pred_contour == 1] = 0

pred_mask = remove_small_objects(label(pred_mask), 4000)
pred_mask[pred_mask != 0] = 255
pred_mask = pred_mask.astype(np.uint8)
plt.imshow(pred_mask)
plt.show()

fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
ax = ax.ravel()
ax[0].imshow(load_img(path, color_mode='rgb'), cmap='gray')
ax[1].imshow(pred_mask, cmap='gray')
ax[2].imshow(pred_contour, cmap='gray')
ax[3].imshow(pred_mask, cmap='gray')

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

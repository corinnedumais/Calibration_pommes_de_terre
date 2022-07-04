# -*- coding: utf-8 -*-
import cv2
import keras
from matplotlib import pyplot as plt

from Utils.utils import show
from Models.Model import dice_loss, combined, dice_coeff
from Utils.segmentation import segment_potatoes

import time

model_mask = keras.models.load_model('Trained Models/mask_+5pics_1ep.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
model_contour = keras.models.load_model('Trained Models/contours_final.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
model_target = keras.models.load_model('Trained Models/targets4.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})


path = 'PDT detection/SolanumTuberosum/Test_images/test15.jpg'
color_img, d, h, mask = segment_potatoes(path, model_mask, model_contour, model_target, patch_size=256, resize=(1024, 768), norm_fact=255)
plt.imshow(mask)
plt.show()
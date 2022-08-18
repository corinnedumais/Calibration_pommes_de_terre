# -*- coding: utf-8 -*-

import os

import random

import numpy as np
from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt
from skimage.filters.rank import modal
from skimage.morphology import rectangle
from tensorflow import keras

from Models.Model import UNetST, dice_loss, dice_coeff, combined
from Utils.segmentation import segment_potatoes, full_prediction
from Utils.target_dimensions import identify_targets
from Utils.utils import show
from dataset_target import CalibrationTargets

train_data_dir = 'Target detection/Dataset Target/Train/Images_patches'
train_target_dir = 'Target detection/Dataset Target/Train/Masks_patches'
eval_data_dir = 'Target detection/Dataset Target/Eval/Images_patches'
eval_target_dir = 'Target detection/Dataset Target/Eval/Masks_patches'

train_input_img_paths = sorted([os.path.join(train_data_dir, file)
                                for file in os.listdir(train_data_dir) if file.endswith(".png")])
train_target_paths = sorted([os.path.join(train_target_dir, file)
                             for file in os.listdir(train_target_dir) if file.endswith(".png")])
val_input_img_paths = sorted([os.path.join(eval_data_dir, file)
                              for file in os.listdir(eval_data_dir) if file.endswith(".png")])
val_target_paths = sorted([os.path.join(eval_target_dir, file)
                           for file in os.listdir(eval_target_dir) if file.endswith(".png")])

batch_size = 8
img_size = (256, 256)
epochs = 50
channels = 8
reg = 0.001
bn = False

model_name = f'target_{channels}c_bs{batch_size}_{epochs}ep_BN{bn}_reg{reg}'

random.Random(1337).shuffle(train_input_img_paths)
random.Random(1337).shuffle(train_target_paths)
random.Random(1337).shuffle(val_input_img_paths)
random.Random(1337).shuffle(val_target_paths)

# Verify shapes
print("Number of samples for training:", len(train_input_img_paths))
print("Number of samples for validation:", len(val_input_img_paths))

# Instantiate data Sequences for each batch
train_gen = CalibrationTargets(batch_size, img_size, train_input_img_paths, train_target_paths)
val_gen = CalibrationTargets(batch_size, img_size, val_input_img_paths, val_target_paths)

# Define callbacks to use during training
callbacks = [keras.callbacks.ModelCheckpoint(f"Trained Models/{model_name}.h5", save_best_only=True),
             keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}')]

# model = UNetST(input_size=(256, 256, 3), output_classes=1, channels=channels, batchnorm=bn, reg=reg).build()
# model.fit(train_gen, batch_size=batch_size, epochs=epochs, validation_data=val_gen, shuffle=True, callbacks=callbacks)

# model = keras.models.load_model('Trained Models/contours_final.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
model = keras.models.load_model(f'Trained Models/{model_name}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
#
# model_mask = keras.models.load_model('Trained Models/mask_yellow.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
# model_contour = keras.models.load_model('Trained Models/cnt_yellow.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
# model_target = keras.models.load_model('Trained Models/targets.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

for i in range(1, 9):
    file_name = f'0{i}.jpg'
    target_im, conversion_factor = identify_targets(model, f'PDT detection/SolanumTuberosum/Test_images/IMG_0101.jpg')
    show(target_im, dims=(800, 600))

### TO CHECK PATCH PREDICTIONS ON VALIDATION DATA ###
# val_preds = model.predict(val_gen)
# for i in range(0, len(val_preds), 25):
#     im = val_preds[i] > 0.5
#     fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
#     ax = axes.ravel()
#
#     ax[0].imshow(load_img(val_input_img_paths[i], color_mode='rgb'), cmap='gray')
#     # ax[0].set_title('Image originale')
#     ax[1].imshow(load_img(val_target_paths[i], color_mode='grayscale'), cmap='gray')
#     # ax[1].set_title('Vérité terrain')
#     ax[2].imshow(im, cmap='gray')
#     # ax[2].set_title('Prédiction du modèle')
#     ax[0].axis('off')
#     ax[1].axis('off')
#     ax[2].axis('off')
#     plt.tight_layout()
#     plt.show()
    # plt.savefig(f'Results/10-03-2022/prediction{i+1:03}')
    # plt.clf()
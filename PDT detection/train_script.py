# -*- coding: utf-8 -*-

import os

import random

import matplotlib.pyplot as plt

from keras.utils import load_img
from tensorflow import keras

from Models.Model import UNetST, dice_loss, dice_coeff
from Utils.dataset import SolanumTuberosum

input_dir = 'PDT detection/SolanumTuberosum/TrainImages'
masks_dir = 'PDT detection/SolanumTuberosum/TrainMasks'
contours_dir = 'PDT detection/SolanumTuberosum/TrainContours'
target_dir = contours_dir

input_img_paths = sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".png")])
target_paths = sorted([os.path.join(target_dir, file) for file in os.listdir(target_dir) if file.endswith(".png")])
assert len(input_img_paths) == len(target_paths), 'Number of targets and images must be equal'

batch_size = 8
img_size = (256, 256)
epochs = 20

# Split our img paths into a training and a validation set
val_samples = 600

random.Random(1339).shuffle(input_img_paths)
random.Random(1339).shuffle(target_paths)
train_input_img_paths, train_target_paths = input_img_paths[:-val_samples], target_paths[:-val_samples]
val_input_img_paths, val_target_paths = input_img_paths[-val_samples:], target_paths[-val_samples:]

# Verify shapes
print("Total number of samples:", len(input_img_paths))
print("Number of samples for training:", len(train_input_img_paths))
print("Number of samples for validation:", len(val_input_img_paths))

# Instantiate data Sequences for each batch
train_gen = SolanumTuberosum(batch_size, img_size, train_input_img_paths, train_target_paths)
val_gen = SolanumTuberosum(batch_size, img_size, val_input_img_paths, val_target_paths)

# Define callbacks to use during training
callbacks = [keras.callbacks.ModelCheckpoint("Trained Models/mask_new_data2.h5", save_best_only=True)]

model = UNetST(input_size=(256, 256, 3), output_classes=1, channels=32).build()
model.fit(train_gen, batch_size=batch_size, epochs=epochs, validation_data=val_gen, shuffle=True, callbacks=callbacks)

# model = keras.models.load_model('Trained Models/contours_final.h5', custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
# model = keras.models.load_model('Trained Models/mask_new_data2.h5', custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

### TO VISUALIZE PREDICTION ON FULL TEST IMAGE ###
# path = 'SolanumTuberosum/Test_images/test4.jpg'
# pred = full_prediction(model, path, 256, (2048, 1536))
# pred = modal(pred, rectangle(3, 3))
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8))
# ax1.imshow(load_img(path, color_mode='rgb'), cmap='gray')
# ax1.axis('off')
# ax2.imshow(pred, cmap='gray')
# ax2.axis('off')
# plt.tight_layout()
# plt.show()

## TO CHECK PATCH PREDICTIONS ON VALIDATION DATA ###
val_preds = model.predict(val_gen)
for i in range(len(val_preds)):
    im = val_preds[i] > 0.5
    fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
    ax = axes.ravel()

    ax[0].imshow(load_img(val_input_img_paths[i], color_mode='rgb'), cmap='gray')
    # ax[0].set_title('Image originale')
    ax[1].imshow(load_img(val_target_paths[i], color_mode='grayscale'), cmap='gray')
    # ax[1].set_title('Vérité terrain')
    ax[2].imshow(im, cmap='gray')
    # ax[2].set_title('Prédiction du modèle')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.tight_layout()
    plt.show()
#     # plt.savefig(f'Results/10-03-2022/prediction{i+1:03}')
#     # plt.clf()

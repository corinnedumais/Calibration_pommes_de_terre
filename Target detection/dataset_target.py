
import os
import random
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.util.shape import view_as_windows
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


class CalibrationTargets(keras.utils.Sequence):
    """
    Helper class to iterate over the dataset to feed batches of data to the model. Code adapted from
    https://keras.io/examples/vision/oxford_pets_image_segmentation/#prepare-sequence-class-to-load-amp-vectorize-batches-of-data
    """

    def __init__(self, batch_size: int, img_size: Tuple[int, int], input_img_paths: List[str], target_paths: List[str], seed=1):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_paths = target_paths
        self.seed = seed

    def __len__(self):
        return len(self.target_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) corresponding to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_paths = self.target_paths[i: i + self.batch_size]

        random.seed(self.seed)
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            # load img in rgb
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        random.seed(self.seed)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, target_path in enumerate(batch_target_paths):
            masks = np.expand_dims(load_img(target_path, target_size=self.img_size, color_mode="grayscale"), -1)
            y[j] = masks

        return x, y


def generate_patches_synthetic(directory: str, window_shape: Tuple[int, int, int], step: int):
    """
    Function that generates patches for all images in the directory. Patches are of shape window_shape and with an
    overlap controlled by the parameter step.
    """
    # Generate patches for all the images
    id_number = 1
    for im_name, mask_name in zip(sorted(os.listdir(os.path.join(directory, 'Images'))),
                                  sorted(os.listdir(os.path.join(directory, 'Masks')))):

        img = cv2.imread(os.path.join(directory, 'Images', im_name))
        img = Image.fromarray(img)
        img = img.resize((2048, 1536), Image.ANTIALIAS)
        img = np.array(img)
        img_crop = view_as_windows(img, window_shape=window_shape, step=step)

        mask = cv2.imread(os.path.join(directory, 'Masks', mask_name), cv2.IMREAD_GRAYSCALE)
        mask = Image.fromarray((mask/255).astype(np.uint8))
        mask = mask.resize((2048, 1536), Image.ANTIALIAS)
        mask = np.array(mask)
        mask_crop = view_as_windows(mask, window_shape=(window_shape[0], window_shape[1]), step=step)

        for i in range(0, img_crop.shape[0]):
            for ii in range(0, img_crop.shape[1]):

                im_saved = img_crop[i, ii, 0, :, :, :] * random.uniform(0.7, 1)
                im90 = np.rot90(im_saved)
                im180 = np.rot90(im90)
                im270 = np.rot90(im180)

                mask_saved = mask_crop[i, ii, :, :]
                mask90 = np.rot90(mask_saved)
                mask180 = np.rot90(mask90)
                mask270 = np.rot90(mask180)

                if np.all((mask_saved == 0)):
                    continue
                else:
                    cv2.imwrite(os.path.join(directory, 'TrainImages2', f'img_{id_number:04}.png'), im_saved)
                    cv2.imwrite(os.path.join(directory, 'TrainImages2', f'img_{id_number+1:04}.png'), im90)
                    cv2.imwrite(os.path.join(directory, 'TrainImages2', f'img_{id_number+2:04}.png'), im180)
                    cv2.imwrite(os.path.join(directory, 'TrainImages2', f'img_{id_number+3:04}.png'), im270)

                    cv2.imwrite(os.path.join(directory, 'TrainMasks2', f'mask_{id_number:04}.png'), mask_saved)
                    cv2.imwrite(os.path.join(directory, 'TrainMasks2', f'mask_{id_number+1:04}.png'), mask90)
                    cv2.imwrite(os.path.join(directory, 'TrainMasks2', f'mask_{id_number+2:04}.png'), mask180)
                    cv2.imwrite(os.path.join(directory, 'TrainMasks2', f'mask_{id_number+3:04}.png'), mask270)

                    id_number += 4


# generate_patches_synthetic('Target detection', window_shape=(256, 256, 3), step=128)

# for i in range(0, 950, 25):
#     fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
#     ax = axes.ravel()
#
#     im = Image.open(f'Target detection/TrainImages/img_{i+1:04}.png')
#     mask = Image.open(f'Target detection/TrainMasks/mask_{i+1:04}.png')
#
#     ax[0].imshow(im)
#     ax[1].imshow(mask)
#     plt.show()
import json
import os
import random
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.draw import polygon
from skimage.io import imread
from skimage.util.shape import view_as_windows
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


class SolanumTuberosum(keras.utils.Sequence):
    """
    Helper class to iterate over the dataset to feed batches of data to the model. Code adapted from
    https://keras.io/examples/vision/oxford_pets_image_segmentation/#prepare-sequence-class-to-load-amp-vectorize-batches-of-data
    """

    def __init__(self, batch_size: int, img_size: Tuple[int, int], input_img_paths: List[str], target_paths: List[str], augment=False, seed=1):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_paths = target_paths
        self.augment = augment
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


def GenerateDataset(directory: str):
    """
    Function to generate the entire dataset. The function takes the original images, retrieves the binary masks (ground
    truth) from the annotations file, resizes both the images and masks and splits them up in patches of shape 256x256.
    """
    # Step 1: Get training images and resize to 3 different sizes
    # for filename in os.listdir(os.path.join(directory, 'Images')):
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         img = Image.open(os.path.join(directory, 'Images', filename))
    #         for size in [(1024, 768), (2048, 1536), (3072, 2304)]:
    #             img = img.resize(size, Image.ANTIALIAS)
    #             img.save(os.path.join(directory, 'Resized_images', f'{filename[:-4]}_{size[0]}_{size[1]}.jpg'), quality=100)

    # Step 2: Get binary masks and contour masks from jason annotations
    # generate_masks(os.path.join(directory, 'Images'), 'annotations.json')
    # generate_contour_maps(os.path.join(directory, 'Images'), 'annotations.json')

    # Step 3: Reduce the binary masks and contour maps to the 3 same sizes as the images
    for filename in os.listdir(os.path.join(directory, 'Masks')):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            mask = Image.open(os.path.join(directory, 'Masks', filename))
            cnt = np.array(Image.open(os.path.join(directory, 'Contours', f'cnt_{filename[5:]}')))
            cnt = cv2.cvtColor(cnt, cv2.COLOR_RGB2GRAY)/255
            cnt = Image.fromarray(cnt.astype(np.uint8))
            for size in [(1024, 768), (2048, 1536), (3072, 2304)]:
                mask = mask.resize(size, Image.ANTIALIAS)
                cnt = cnt.resize(size, Image.ANTIALIAS)

                mask.save(os.path.join(directory, 'Resized_masks', f'mask_{filename[:-4]}_{size[0]}_{size[1]}.png'), quality=100)
                cnt.save(os.path.join(directory, 'Resized_contours', f'cnt_{filename[5:-4]}_{size[0]}_{size[1]}.png'), quality=100)

    # Step 4: Generate the patches (images, masks and contours) for training
    generate_patches(directory, window_shape=(256, 256, 3), step=128)


def generate_contour_maps(directory: str, annotations_file: str):
    annotations = json.load(open(os.path.join(directory, annotations_file)))
    annotations = list(annotations.values())

    for a in annotations:
        if a['regions']:
            polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(directory, a['filename'])
            image = imread(image_path)
            height, width = image.shape[:2]

            mask = np.zeros([height, width], dtype=np.uint8)
            contours = []
            for p in polygons:
                y, x = p['all_points_y'], p['all_points_x']
                contour = np.expand_dims(list(zip(x, y)), 1)
                contours.append(contour)

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), 7)
            cv2.imwrite(f"SolanumTuberosum/Contours/cnt_{a['filename'][:-4]}.png", mask)
            if f"{a['filename'][:2]}_.jpg" in os.listdir(directory):
                cv2.imwrite(f"SolanumTuberosum/Contours/cnt_{a['filename'][:2]}_.png", mask)


def generate_masks(root_dir: str, annotations_file: str):
    """
    Function to generate the binary masks from json annotation files. Saves the masks as PNG files in 'Masks' subdirectory.
    """
    annotations = json.load(open(os.path.join(root_dir, annotations_file)))
    annotations = list(annotations.values())

    for a in annotations:
        if a['regions']:
            polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(root_dir, a['filename'])
            image = imread(image_path)
            height, width = image.shape[:2]

            mask = np.zeros([height, width], dtype=np.uint8)
            for p in polygons:
                x, y = p['all_points_y'], p['all_points_x']
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = polygon(x, y)
                mask[rr, cc] = 1

            cv2.imwrite(f"SolanumTuberosum/Masks/mask_{a['filename'][:2]}.png", mask)
            if f"{a['filename'][:2]}_.jpg" in os.listdir(root_dir):
                cv2.imwrite(f"SolanumTuberosum/Masks/mask_{a['filename'][:2]}_.png", mask)


def generate_patches(directory: str, window_shape: Tuple[int, int, int], step: int):
    """
    Function that generates patches for all images in the directory. Patches are of shape window_shape and with an
    overlap controlled by the parameter step.
    """
    # Generate patches for all the images
    id_number = 1
    for im_name, mask_name, contour_name in zip(sorted(os.listdir(os.path.join(directory, 'Resized_images'))),
                                                sorted(os.listdir(os.path.join(directory, 'Resized_masks'))),
                                                sorted(os.listdir(os.path.join(directory, 'Resized_contours')))):

        img = cv2.imread(os.path.join(directory, 'Resized_images', im_name))
        img_crop = view_as_windows(img, window_shape=window_shape, step=step)

        mask = cv2.imread(os.path.join(directory, 'Resized_masks', mask_name), cv2.IMREAD_GRAYSCALE)
        mask_crop = view_as_windows(mask, window_shape=(window_shape[0], window_shape[1]), step=step)

        contour = cv2.imread(os.path.join(directory, 'Resized_contours', contour_name), cv2.IMREAD_GRAYSCALE)
        contour_crop = view_as_windows(contour, window_shape=(window_shape[0], window_shape[1]), step=step)

        assert img_crop.shape[:2] == mask_crop.shape[:2]

        for i in range(0, img_crop.shape[0]):
            for ii in range(0, img_crop.shape[1]):

                im_file = os.path.join(directory, 'TrainImages', f'img_{id_number:04}.png')
                im_saved = img_crop[i, ii, 0, :, :, :] * random.uniform(0.7, 1)

                contour_file = os.path.join(directory, 'TrainContours', f'cnt_{id_number:04}.png')
                contour_saved = contour_crop[i, ii, :, :]

                mask_file = os.path.join(directory, 'TrainMasks', f'mask_{id_number:04}.png')
                mask_saved = mask_crop[i, ii, :, :]

                if np.all((mask_saved == 0)):
                    continue
                else:
                    cv2.imwrite(im_file, im_saved)
                    cv2.imwrite(mask_file, mask_saved)
                    cv2.imwrite(contour_file, contour_saved)
                    id_number += 1


def flip_im(img_name):
    im = Image.open(os.path.join('SolanumTuberosum', 'Images', img_name))
    im = np.rot90(im)
    # im = np.fliplr(np.array(im))
    im = Image.fromarray(im)
    im.save(os.path.join('SolanumTuberosum', 'Images', img_name))


# GenerateDataset('SolanumTuberosum')
#
# dir_pdt = 'SolanumTuberosum'
#
# for im_name, mask_name, contour_name in zip(sorted(os.listdir(os.path.join(dir_pdt, 'Resized_images')))[::3],
#                                             sorted(os.listdir(os.path.join(dir_pdt, 'Resized_masks')))[::3],
#                                             sorted(os.listdir(os.path.join(dir_pdt, 'Resized_contours')))[::3]):
#     fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 5))
#     ax = axes.ravel()
#
#     im = Image.open(os.path.join(dir_pdt, 'Resized_images', im_name))
#     mask = Image.open(os.path.join(dir_pdt, 'Resized_masks', mask_name))
#     cnt = Image.open(os.path.join(dir_pdt, 'Resized_contours', contour_name))
#
#     ax[0].imshow(im)
#     ax[1].imshow(mask)
#     ax[2].imshow(cnt)
#     plt.show()
#
# for i in range(0, 3000, 50):
#     fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 5))
#     ax = axes.ravel()
#
#     im = Image.open(f'SolanumTuberosum/TrainImages/img_{i+1:04}.png')
#     mask = Image.open(f'SolanumTuberosum/TrainMasks/mask_{i+1:04}.png')
#     cnt = Image.open(f'SolanumTuberosum/TrainContours/cnt_{i+1:04}.png')
#
#     ax[0].imshow(im)
#     ax[1].imshow(mask)
#     ax[2].imshow(cnt)
#     plt.show()

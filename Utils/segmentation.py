# -*- coding: utf-8 -*-
import time
import warnings
from typing import Tuple, Any, List, Union

import cv2
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects, remove_small_holes

from Utils.gapfilling import fill_gaps

warnings.filterwarnings('ignore', category=UserWarning)


def full_prediction(trained_model, img, patch_size: int, resize: Tuple[int, int], norm_fact) -> np.ndarray:
    """
    Function to make a full prediction on an image.

    Resizes the image, divises the image in patches, makes a prediction for each patch and reassembles the full image.

    Parameters
    ----------
    trained_model
                  Trained model used for the patch predictions
    img: str or PIL image
              Image
    patch_size: int
                Size of the (square) patch to use, should match the input of the trained model.
    resize: tuple of int
            Size to resize image
    norm_fact: int
            Normalization factor used on training images for the model used for inference

    Returns
    -------
    np.ndarray
               Prediction for the entire image.
    """
    assert resize[0] % patch_size == 0 and resize[1] % patch_size == 0

    # Open and resize image
    if isinstance(img, str):
        img = Image.open(img)
    image = img.resize(resize, Image.ANTIALIAS)
    image = np.array(image) / norm_fact

    # Predict each patch and reassemble full image
    segm_img = np.expand_dims(np.zeros(image.shape[:2]), -1)  # Array with zeros to be filled with segmented values
    s_time = time.time()
    for i in range(0, image.shape[0], patch_size):  # Steps of 256
        for j in range(0, image.shape[1], patch_size):  # Steps of 256
            single_patch = np.expand_dims(image[i:i + patch_size, j:j + patch_size], axis=0)
            single_patch_prediction = (trained_model.predict(single_patch, verbose=0) > 0.5).astype(np.uint8)[0, :, :,
                                      :]
            segm_img[i:i + patch_size, j:j + patch_size, :] += single_patch_prediction
    print(f'(inference time: {time.time() - s_time:.2f} s)', end=' ')

    return segm_img[:, :, 0]


def mm_per_pixel(target_model, img: str, resize: Tuple[int, int], norm_fact: int) -> Tuple[float, list, list]:
    """
    Function to find the targets and return infos.

    Parameters
    ----------
    target_model
            Trained model for target detection
    img: str or PIL image
            Image or image path
    resize: tuple of int
            Size to resize image
    norm_fact: int
            Normalization factor used on training images for the model used for inference

    Returns
    -------
    mm_per_px: float
            Average conversion factor
    cnt: list
         All contour boxes for the targets
    target_infos: list of dict
         Position and dimension of each target
    """
    # Get target prediction
    pred = full_prediction(target_model, img=img, patch_size=256, resize=resize, norm_fact=norm_fact)
    pred = modal(img_as_ubyte(pred), rectangle(5, 5))
    pred = remove_small_objects(label(pred), 1500)
    pred[pred != 0] = 255
    pred = pred.astype(np.uint8)

    # Get contours
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes, cnt, target_infos = [], [], []

    for contour in contours:
        # Fit rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1][0], rect[1][1]
        # Check if ressembles enough a square
        if 0.75 < width / height < 1.25:
            min_dim = np.amin([width, height])
            sizes.append(min_dim)
            rect = (rect[0], (min_dim, min_dim), rect[2])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cnt.append(box)
            target_infos.append({'pos': rect[0], 'dim': min_dim})

    # Mean size of all detected targets
    mean_size = np.mean(sizes)
    mm_per_px = 40 / mean_size
    return mm_per_px, cnt, target_infos


def segment_potatoes(img, mask_model, contours_model, target_model, patch_size: int, resize: Tuple[int, int],
                     norm_fact, variety='burbank', dev_mode=True) -> Tuple[Any, list, list, int, Any]:
    """
    Function to segment the potatoes from an image using the models trained with UNet architecture.

    The function resizes the image and fractions it into tiles of 256x256, then predicts both mask and contours on all
    tiles and reassembles the full mask and contours predictions. Contours are subtracted from mask and all individual
    remaining shapes are considered a potato. An ellipse is drawn around each with cv2 functions.

    Parameters
    ----------
    img: str or PIL image
              Image
    mask_model
               Trained model for mask semantic segmentation
    contours_model
                   Trained model for contour identification
    target_model
                   Trained model for target identification
    patch_size: int
                Size of the (square) patch to use, should match the input of the trained model
    resize: tuple of int
            Size to resize image
    norm_fact: int
              Factor by which to normalize the image to pass to the trained model
    variety: str
              Variety of potatoes, should be 'burbank' or 'mountain_gem'
    dev_mode: bool
              Development mode displays ellipses and rectangles around object. If False, only red 'x' are displayed.

    Returns
    -------
    color_img: np.ndarray
               The original RGB image with the drawn ellipses.
    diameters: list
               List of all the objects' diameters
    heights: list
             List of all the objects' heights
    target: int
            Number of targets detected
    mask: np.ndarray
         copy of segmentation mask
    """

    # Analyze targets
    print('Analyzing targets...', end=' ')
    s_time = time.time()
    av_conv_factor, target_cnt, target_infos = mm_per_pixel(target_model, img, resize, norm_fact)
    print(f'Terminated after {time.time() - s_time: .2f} s')

    # Mask predictions and apply modal filter to eliminate artifacts at the junction of the predicted tiles
    print('Getting mask prediction...', end=' ')
    s_time = time.time()
    pred_mask = full_prediction(mask_model, img, patch_size, resize, norm_fact=norm_fact)
    pred_mask = modal(img_as_ubyte(pred_mask), rectangle(5, 5))
    print(f'Terminated after {time.time() - s_time: .2f} s')

    # Contour predictions
    print('Getting contours prediction...', end=' ')
    s_time = time.time()
    pred_contour = full_prediction(contours_model, img, patch_size, resize, 255)
    pred_contour = modal(img_as_ubyte(pred_contour), rectangle(5, 5))
    print(f'Terminated after {time.time() - s_time: .2f} s')

    print('Applying post-processing steps...', end=' ')
    s_time = time.time()

    # Remove small object from mask
    pred_mask = remove_small_objects(label(pred_mask), 1500)
    pred_mask[pred_mask != 0] = 255

    # Thin contour and remove small objects
    pred_contour = cv2.ximgproc.thinning(pred_contour) / 255
    pred_contour = pred_contour.astype(np.uint8)
    pred_contour = remove_small_objects(label(pred_contour), 50)
    pred_contour[pred_contour != 0] = 1

    pred_contour = fill_gaps(pred_contour, n_iterations=1)  # Uncomment to use gap-filling algorithm
    pred_contour[pred_contour != 0] = 1

    # Substruction of contours
    pred_contour = pred_contour.astype(np.float32)
    pred_contour = cv2.dilate(pred_contour, np.ones((3, 3)))
    pred_contour = pred_contour.astype(np.uint8)
    pred_mask[pred_contour == 1] = 0

    # Second clean-up: remove again small objects or holes create by the subtraction
    pred_mask = remove_small_objects(label(pred_mask), 1500)
    pred_mask[pred_mask != 0] = 255
    pred_mask = remove_small_holes(label(pred_mask), 4000)
    pred_mask[pred_mask != 0] = 255
    pred_mask = pred_mask.astype(np.uint8) * 255

    mask = pred_mask.copy()  # For development purposes

    # Load the image in RGB
    if isinstance(img, str):
        img = Image.open(img)
    color_img = img.resize(resize, Image.ANTIALIAS)
    color_img = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)
    print(f'Terminated after {time.time() - s_time: .2f} s')

    print('Getting dimensions and drawing result...', end=' ')
    s_time = time.time()
    contours, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lists to store dimensions
    widths, heights = [], []

    for contour in contours:
        # The contour needs to be made of at least 5 points to fit an ellipse
        if len(contour) > 4:
            ellipse = cv2.fitEllipse(contour)

            # Center coordinates (must be integers)
            xc, yc = round(ellipse[0][0]), round(ellipse[0][1])
            angle = ellipse[2]

            # Rotate shape
            angle = -angle if angle < 90 else 180 - angle
            rotated = rotate(np.squeeze(contour), (xc, yc), angle)
            x_sorted = sorted(rotated, key=lambda x: x[0])
            y_sorted = sorted(rotated, key=lambda x: x[1])
            WIDTH = x_sorted[-1][0] - x_sorted[0][0]
            HEIGHT = y_sorted[-1][1] - y_sorted[0][1]

            # Find closest target
            if len(list(target_cnt)) == 0:
                conv_factor = 1
                warnings.warn('No targets were detected. Measurements will be given in pixels.')
            else:
                sorted_targets = sorted(target_infos,
                                        key=lambda x: np.sqrt((x['pos'][0] - xc) ** 2 + (x['pos'][1] - yc) ** 2))
                conv_factor = 40 / sorted_targets[0]['dim']

            # Semi-axis (must be integers)
            widthE, heightE = ellipse[1]

            # Correction factor to account for contour subtraction
            correction = 4

            # Ellipse params
            a, b = round(0.5 * heightE + correction / 2), round(0.5 * widthE + correction / 2)

            # Corrected measurements in pixels
            width_px = WIDTH + correction if variety == 'burbank' else WIDTH - correction
            height_px = HEIGHT + correction if variety == 'burbank' else HEIGHT - correction

            # Measurements in pixels
            width_mm, height_mm = width_px * conv_factor, height_px * conv_factor

            # we filter the ellipses to eliminate those who are too small
            if 35 < width_mm < 150:
                if dev_mode:
                    cv2.ellipse(color_img, (xc, yc), (b, a), ellipse[2], 0, 360, (0, 0, 255), 3)
                    rect = (ellipse[0], (width_px, height_px), ellipse[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(color_img, [box], 0, (0, 255, 255), 3)

                    # # Uncomment to display measurements on result image
                    # cv2.putText(color_img, f'{width_mm:.0f}x{height_mm:.0f}', (xc - 40, yc),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                else:
                    cv2.putText(color_img, 'x', (xc - 20, yc + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                                cv2.LINE_AA)

                widths.append(width_mm)
                heights.append(height_mm)

            for box in target_cnt:
                # Draw contour of the target
                cv2.drawContours(color_img, [box], 0, (0, 255, 0), 3)

    print(f'Terminated after {time.time() - s_time: .2f} s')

    return color_img, widths, heights, len(list(target_cnt)), mask


def rotate(points, center, angle):
    """
    Utility function to rotate a set of points of a given angle around a given center.
    """
    ANGLE = np.deg2rad(angle)
    c_x, c_y = center
    return np.array([[c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                      c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)]
                     for px, py in points])

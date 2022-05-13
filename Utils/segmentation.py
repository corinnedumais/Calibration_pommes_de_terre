# -*- coding: utf-8 -*-
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.filters.rank import modal
from skimage.morphology import rectangle

from Utils.utils import pixels_to_mm


def full_prediction(trained_model, img_path: str, patch_size: int, resize: Tuple[int, int]) -> np.ndarray:
    """
    Function to make a full prediction on an image.

    Resizes the image, divises the image in patches, makes a prediction for each patch and reassembles the full image.

    Parameters
    ----------
    trained_model
                  Trained model used for the patch predictions
    img_path: str
              Path to the image.
    patch_size: int
                Size of the (square) patch to use, should match the input of the trained model.
    resize: tuple of int
            Size to resize image

    Returns
    -------
    np.ndarray
               Prediction for the entire image.
    """
    assert resize[0] % patch_size == 0 and resize[1] % patch_size == 0

    # Open and resize image
    img = Image.open(img_path)
    image = img.resize(resize, Image.ANTIALIAS)
    image = np.array(image)

    # Predict each patch and reassemble full image
    segm_img = np.expand_dims(np.zeros(image.shape[:2]), -1)  # Array with zeros to be filled with segmented values
    for i in range(0, image.shape[0], patch_size):  # Steps of 256
        for j in range(0, image.shape[1], patch_size):  # Steps of 256
            single_patch = np.expand_dims(image[i:i + patch_size, j:j + patch_size], axis=0)
            single_patch_prediction = (trained_model.predict(single_patch) > 0.5).astype(np.uint8)[0, :, :, :]
            segm_img[i:i + patch_size, j:j + patch_size, :] += single_patch_prediction

    return segm_img[:, :, 0]


def segment_potatoes(img_path: str, mask_model, contours_model, patch_size: int, resize: Tuple[int, int]) -> Tuple[np.ndarray, list, list]:
    """
    Function to segment the potatoes from an image using the models trained with UNet architecture.

    The function resizes the image and fractions it into tiles of 256x256, then predicts both mask and contours on all
    tiles and reassembles the full mask and contours predictions. Contours are subtracted from mask and all individual
    remaining shapes are considered a potato. An ellipse is drawn around each with cv2 functions.

    Parameters
    ----------
    img_path: str
              Path of the image for the segmentation
    mask_model
               Trained model for mask semantic segmentation
    contours_model
                   Trained model for contour identification
    patch_size: int
                Size of the (square) patch to use, should match the input of the trained model
    resize: tuple of int
            Size to resize image

    Returns
    -------
    color_img: np.ndarray
               The original RGB image with the drawn ellipses.
    diameters: list
               List of all the objects' diameters
    heights: list
             List of all the objects' heights
    """
    # Mask and contour predictions
    pred_mask = full_prediction(mask_model, img_path, patch_size, resize)
    pred_contour = full_prediction(contours_model, img_path, patch_size, resize)

    # Modal filter to eliminate artifacts at the junction of the predicted tiles
    pred_mask = modal(pred_mask, rectangle(5, 5))
    pred_contour = modal(pred_contour, rectangle(5, 5))

    # Subtraction of the two masks and elimination of negative values
    pred = pred_mask - pred_contour
    pred[pred < 5] = 0

    # Load the image in RGB
    color_img = Image.open(img_path)
    color_img = color_img.resize(resize, Image.ANTIALIAS)
    color_img = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)

    # Locate the edges
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lists to store dimensions
    widths, heights = [], []

    for contour in contours:
        # The contour needs to be made of at least 5 points to fit an ellipse
        if len(contour) > 4:
            ellipse = cv2.fitEllipse(contour)

            # Center coordinates (must be integers)
            xc, yc = round(ellipse[0][0]), round(ellipse[0][1])

            # Factors to account for erosion due to contour subtraction.
            factor_h = 1.05
            factor_w = 1.1

            # Semi-axis (must be integers)
            widthE, heightE = ellipse[1]
            a, b = round(0.5 * heightE * factor_h), round(0.5 * widthE * factor_w)
            angle = ellipse[2]

            # we filter the ellipses to eliminate those who are too small
            if a > 10 and b > 10:
                cv2.ellipse(color_img, (xc, yc), (b, a), angle, 0, 360, (0, 0, 255), 2)
                cv2.putText(color_img, f'{pixels_to_mm(widthE * factor_w, 72, 1.97):.0f} mm', (xc - 25, yc),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 205, 50), 2, cv2.LINE_AA)
                cv2.putText(color_img, f'{pixels_to_mm(heightE * factor_h, 72, 1.97):.0f} mm', (xc - 25, yc + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2, cv2.LINE_AA)
                widths.append(widthE * factor_w)
                heights.append(heightE * factor_h)

            cv2.imwrite('preds.png', color_img)

    return color_img, widths, heights

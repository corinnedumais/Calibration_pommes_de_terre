# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters.rank import modal
from skimage.morphology import rectangle

from Utils.conversion import pixels_to_mm

def full_prediction(trained_model, img_path, patch_size, resize):
    """
    Function to make a full prediction on an image. Resizes the image, divises the image in patches, makes a prediction
    for each patch and reassembles the full image.

    Parameters:
        trained_model: Trained model used for the patch predictions
        img_path (str): Path for the image
        patch_size (int): Size of the (square) patch to use, should match the input of the trained model
        resize (tuple): Size to resize image

    Returns the full segmentation prediction.
    """
    assert resize[0] % patch_size == 0 and resize[1] % patch_size == 0

    img = Image.open(img_path)
    image = img.resize(resize, Image.ANTIALIAS)
    image = np.array(image)
    segm_img = np.expand_dims(np.zeros(image.shape[:2]), -1)  # Array with zeros to be filled with segmented values
    for i in range(0, image.shape[0], patch_size):  # Steps of 256
        for j in range(0, image.shape[1], patch_size):  # Steps of 256
            single_patch = np.expand_dims(image[i:i + patch_size, j:j + patch_size], axis=0)
            single_patch_prediction = (trained_model.predict(single_patch) > 0.5).astype(np.uint8)[0, :, :, :]
            segm_img[i:i + patch_size, j:j + patch_size, :] += single_patch_prediction
    return segm_img[:, :, 0]


def segment_potatoes(img_path, mask_model, contours_model, patch_size, resize):
    """
    Function to segment the potatoes from an image using the models trained with UNet architecture.

    The function resizes the image and fractions it into tiles of 256x256, then predicts both mask and contours on all
    tiles and reassembles the full mask and contours predictions. Contours are subtracted from mask and all individual
    remaining shapes are considered a potato. Ellipse is draw around each with cv2 functions.

    Parameters:
        img_path (str): Path of the image for the segmentation
        mask_model: Trained model for mask semantic segmentation
        contours_model: Trained model for contour identification
        patch_size (int): Size of the (square) patch to use, should match the input of the trained model
        resize (tuple): Size to resize image

    Returns:
        color_img: The RGB image with the drawn ellipses
        diameters (list): List of all the objects' diameters
        heights (list): List of all the objects' heights
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
    widths, heights, angles = [], [], []

    for contour in contours:
        # The contour needs to be made of at least 5 points to fit an ellipse
        if len(contour) > 4:
            # Use the ellipse fit to compute the center coordinates and the angle
            ellipse = cv2.fitEllipse(contour)
            # Center coordinated must be integers
            factor_h = 1.05
            factor_w = 1.1
            xc, yc = round(ellipse[0][0]), round(ellipse[0][1])
            widthE, heightE = ellipse[1]
            a, b = round(0.5 * heightE * factor_h), round(0.5 * widthE * factor_w)
            angle = ellipse[2]

            # we filter the ellipses to eliminate those who are too small
            if a > 10 and b > 10:
                rect = cv2.minAreaRect(contour)
                if rect[1][0] > rect[1][1]:
                    widthR, heightR = rect[1][1], rect[1][0]
                else:
                    widthR, heightR = rect[1][0], rect[1][1]

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # cv2.drawContours(color_img, [box], 0, (0, 0, 205), 2)
                cv2.ellipse(color_img, (xc, yc), (b, a), angle, 0, 360, (0, 0, 255), 2)
                if True:
                    cv2.putText(color_img, f'{pixels_to_mm(widthE * factor_w, angle, 72, 1.97):.0f} mm', (xc-25, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 205, 50), 2, cv2.LINE_AA)
                    cv2.putText(color_img, f'{pixels_to_mm(heightE * factor_h, angle, 72, 1.97):.0f} mm', (xc-25, yc+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2, cv2.LINE_AA)
                widths.append(widthE * factor_w)
                heights.append(heightE * factor_h)
                angles.append(angle)

            cv2.imwrite('preds.png', color_img)

    return color_img, widths, heights, angles

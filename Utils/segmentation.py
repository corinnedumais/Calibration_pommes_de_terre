# -*- coding: utf-8 -*-
import time
from typing import Tuple

import cv2
import numpy as np
import skimage.draw
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects, remove_small_holes

from Utils.gapfilling import fill_gaps


def full_prediction(trained_model, img_path: str, patch_size: int, resize: Tuple[int, int], norm_fact) -> np.ndarray:
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
    image = np.array(image)/norm_fact

    # Predict each patch and reassemble full image
    segm_img = np.expand_dims(np.zeros(image.shape[:2]), -1)  # Array with zeros to be filled with segmented values
    for i in range(0, image.shape[0], patch_size):  # Steps of 256
        for j in range(0, image.shape[1], patch_size):  # Steps of 256
            single_patch = np.expand_dims(image[i:i + patch_size, j:j + patch_size], axis=0)
            single_patch_prediction = (trained_model.predict(single_patch, verbose=0) > 0.5).astype(np.uint8)[0, :, :, :]
            segm_img[i:i + patch_size, j:j + patch_size, :] += single_patch_prediction

    return segm_img[:, :, 0]


def mm_per_pixel(target_model, img_path, norm_fact):
    # Get target prediction
    pred = full_prediction(target_model, img_path=img_path, patch_size=256, resize=(2048, 1536), norm_fact=1)
    pred = modal(pred, rectangle(5, 5))
    pred = remove_small_objects(label(pred), 1500)
    pred[pred != 0] = 255
    pred = pred.astype(np.uint8)

    plt.imshow(pred)
    plt.show()

    # Get contours
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes = []
    cnt = []
    for contour in contours:
        # Fit rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1][0], rect[1][1]
        # Check if ressembles enough a square
        if 0.85 < width / height < 1.15:
            sizes.append(np.amin([width, height]))
            cnt.append(contour)

    # Mean size of all detected targets
    mean_size = np.mean(sizes)
    mm_per_px = 40 / mean_size
    return mm_per_px, cnt


def segment_potatoes(img_path: str, mask_model, contours_model, target_model, patch_size: int, resize: Tuple[int, int],
                     norm_fact) -> Tuple[np.ndarray, list, list]:
    """
    Function to segment the potatoes from an image using the models trained with UNet architecture.

    The function resizes the image and fractions it into tiles of 256x256, then predicts both mask and contours on all
    tiles and reassembles the full mask and contours predictions. Contours are subtracted from mask and all individual
    remaining shapes are considered a potato. An ellipse is drawn around each with cv2 functions.

    Parameters
    ----------
    norm_fact: int
              Factor by which to normalize the image to pass to the trained model
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
    # conv_factor, target_cnt = mm_per_pixel(target_model, img_path, norm_fact)
    conv_factor, target_cnt = 1, []

    # Mask and contour predictions
    pred_mask = full_prediction(mask_model, img_path, patch_size, resize, norm_fact=norm_fact)
    # pred_contour = full_prediction(contours_model, img_path, patch_size, resize, 255)

    # Modal filter to eliminate artifacts at the junction of the predicted tiles
    pred_mask = modal(pred_mask, rectangle(7, 7))
    # pred_contour = modal(pred_contour, rectangle(5, 5))
    plt.imshow(pred_mask)
    plt.axis('off')
    plt.show()

    pred_mask = cv2.dilate(pred_mask, np.ones((3, 3), np.uint8), iterations=1)
    pred_mask = remove_small_objects(label(pred_mask), 2000)
    pred_mask[pred_mask != 0] = 255

    pred_mask = remove_small_holes(label(pred_mask), 3000)
    pred_mask[pred_mask != 0] = 255
    pred_mask = pred_mask.astype(np.uint8) * 255

    ### WATERSHED ###
    inverse = cv2.bitwise_not(pred_mask)
    skeleton = cv2.ximgproc.thinning(inverse)/255
    gap_fill = fill_gaps(skeleton, 5, display_all_it=False)
    pred_mask[gap_fill != 0] = 0
    pred_mask = cv2.erode(pred_mask, np.ones((3, 3), np.uint8), iterations=1)

    inverse[inverse == 0] = 0.5
    inverse[skeleton != 0] = 0

    fig, axes = plt.subplots(ncols=2, figsize=(12, 7))
    ax = axes.ravel()
    ax[0].imshow(inverse)
    ax[1].imshow(pred_mask)
    plt.show()
    #################

    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # ax1.imshow(pred_mask)
    #
    # pred_contour = cv2.ximgproc.thinning(pred_contour)/255
    # pred_contour = pred_contour.astype(np.uint8)
    #
    # pred_contour = remove_small_objects(label(pred_contour), 50)
    # pred_contour[pred_contour != 0] = 1

    ###############################

    # start_time = time.time()
    # pred_contour = fill_gaps(pred_contour, n_iterations=12)
    # print(f'Execution time: {time.time() - start_time: .3f} seconds')

    ###############################################

    # pred_contour[pred_contour != 0] = 1
    # pred_contour = pred_contour.astype(np.float32)
    # pred_contour = cv2.dilate(pred_contour, np.ones((3, 3)))
    # pred_contour = pred_contour.astype(np.uint8)

    # Subtraction of the two masks and elimination of negative values
    # pred_mask[pred_contour == 1] = 0

    # ax2.imshow(pred_mask)
    # plt.tight_layout()
    # plt.show()

    # Load the image in RGB
    color_img = Image.open(img_path)
    color_img = color_img.resize(resize, Image.ANTIALIAS)
    color_img = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)

    contours, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

    # Locate the edges
    contours, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Lists to store dimensions
    widths, heights = [], []

    for contour in contours:
        # The contour needs to be made of at least 5 points to fit an ellipse
        if len(contour) > 4:
            ellipse = cv2.fitEllipse(contour)

            # Center coordinates (must be integers)
            xc, yc = round(ellipse[0][0]), round(ellipse[0][1])

            # Semi-axis (must be integers)
            widthE, heightE = ellipse[1]
            a, b = round(0.5 * heightE) + 4, round(0.5 * widthE) + 4
            angle = ellipse[2]

            width = widthE * conv_factor
            height = heightE * conv_factor

            # we filter the ellipses to eliminate those who are too small
            if width > 25 and height > 25:
                cv2.ellipse(color_img, (xc, yc), (b, a), angle, 0, 360, (0, 0, 255), 2)
                # cv2.putText(color_img, f'{width:.0f} mm', (xc - 25, yc),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 205, 50), 2, cv2.LINE_AA)
                # cv2.putText(color_img, f'{height:.0f} mm', (xc - 25, yc + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                widths.append(width)
                heights.append(height)

            for tc in target_cnt:
                rect = cv2.minAreaRect(tc)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # Draw contour of the target
                cv2.drawContours(color_img, [box], 0, (0, 128, 255), 2)

            cv2.imwrite('preds.png', color_img)

    return color_img, widths, heights

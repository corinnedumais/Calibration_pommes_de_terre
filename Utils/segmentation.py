# -*- coding: utf-8 -*-
import time
from typing import Tuple, Any, List, Union

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
    image = np.array(image)/norm_fact

    # Predict each patch and reassemble full image
    segm_img = np.expand_dims(np.zeros(image.shape[:2]), -1)  # Array with zeros to be filled with segmented values
    for i in range(0, image.shape[0], patch_size):  # Steps of 256
        for j in range(0, image.shape[1], patch_size):  # Steps of 256
            single_patch = np.expand_dims(image[i:i + patch_size, j:j + patch_size], axis=0)
            single_patch_prediction = (trained_model.predict(single_patch, verbose=0) > 0.5).astype(np.uint8)[0, :, :, :]
            segm_img[i:i + patch_size, j:j + patch_size, :] += single_patch_prediction

    return segm_img[:, :, 0]


def mm_per_pixel(target_model, img_path, resize, norm_fact):
    # Get target prediction
    pred = full_prediction(target_model, img=img_path, patch_size=256, resize=resize, norm_fact=norm_fact)
    pred = modal(pred, rectangle(5, 5))
    pred = remove_small_objects(label(pred), 1500)
    pred[pred != 0] = 255
    pred = pred.astype(np.uint8)

    # plt.imshow(pred)
    # plt.show()

    # Get contours
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes, cnt, target_infos = [], [], []

    for contour in contours:
        # Fit rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1][0], rect[1][1]
        # Check if ressembles enough a square
        if 0.85 < width / height < 1.15:
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
                     norm_fact, dev_mode=True) -> Tuple[Any, List[Union[int, Any]], List[Union[int, Any]], int]:
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
    patch_size: int
                Size of the (square) patch to use, should match the input of the trained model
    resize: tuple of int
            Size to resize image
    norm_fact: int
              Factor by which to normalize the image to pass to the trained model
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
    """
    conv_factor, target_cnt, target_infos = mm_per_pixel(target_model, img, resize, norm_fact)
    # conv_factor, target_cnt = 1, []
    if len(target_cnt) == 0:
        conv_factor = 1
        raise(Warning('No targets were detected.'))

    # Mask and contour predictions
    pred_mask = full_prediction(mask_model, img, patch_size, resize, norm_fact=norm_fact)
    pred_contour = full_prediction(contours_model, img, patch_size, resize, 255)

    # Modal filter to eliminate artifacts at the junction of the predicted tiles
    pred_mask = modal(pred_mask, rectangle(7, 7))
    pred_contour = modal(pred_contour, rectangle(5, 5))

    pred_mask = cv2.dilate(pred_mask, np.ones((3, 3), np.uint8), iterations=1)
    pred_mask = remove_small_objects(label(pred_mask), 1500)
    pred_mask[pred_mask != 0] = 255

    pred_contour = cv2.ximgproc.thinning(pred_contour)/255
    pred_contour = pred_contour.astype(np.uint8)

    pred_contour = remove_small_objects(label(pred_contour), 50)
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
    pred_mask = remove_small_objects(label(pred_mask), 1000)
    pred_mask[pred_mask != 0] = 255

    pred_mask = remove_small_holes(label(pred_mask), 3000)
    pred_mask[pred_mask != 0] = 255
    pred_mask = pred_mask.astype(np.uint8) * 255

    # ax2.imshow(pred_mask)
    # plt.tight_layout()
    # plt.show()

    # Load the image in RGB
    if isinstance(img, str):
        img = Image.open(img)
    color_img = img.resize(resize, Image.ANTIALIAS)
    color_img = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)

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
            sorted_targets = sorted(target_infos, key=lambda x: np.sqrt((x['pos'][0] - xc)**2 + (x['pos'][1] - yc)**2))
            conv_factor = 40/sorted_targets[0]['dim']

            # Semi-axis (must be integers)
            widthE, heightE = ellipse[1]

            # Ellipse params
            a, b = round(0.5 * heightE) + 4, round(0.5 * widthE) + 4

            width_px = widthE + 8 if widthE < WIDTH else WIDTH + 8
            height_px = heightE + 8 if heightE < HEIGHT else HEIGHT + 8

            width_mm, height_mm = width_px * conv_factor, height_px * conv_factor

            # we filter the ellipses to eliminate those who are too small
            if 25 < width_mm < 150:
                if dev_mode:
                    cv2.ellipse(color_img, (xc, yc), (b, a), ellipse[2], 0, 360, (255, 255, 0), 2)
                    rect = (ellipse[0], (width_px, height_px), ellipse[2])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(color_img, [box], 0, (0, 255, 255), 2)

                    cv2.putText(color_img, f'{width_mm:.0f}x{height_mm:.0f}', (xc - 40, yc),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(color_img, 'x', (xc-20, yc+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                # cv2.putText(color_img, f'{(WIDTH + 8) * conv_factor:.0f}x{(HEIGHT + 8) * conv_factor:.0f}', (xc - 25, yc + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                widths.append(width_mm)
                heights.append(height_mm)

            for box in target_cnt:
                # Draw contour of the target
                cv2.drawContours(color_img, [box], 0, (0, 255, 0), 3)

            # cv2.imwrite('preds.png', color_img)

    return color_img, widths, heights, len(target_cnt)


def rotate(points, center, angle):
    ANGLE = np.deg2rad(angle)
    c_x, c_y = center
    return np.array(
        [
            [
                c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
            ]
            for px, py in points
        ]
    )


def calibrate(img_path: str, mask_model, contours_model, target_model, patch_size: int, resize: Tuple[int, int],
                     norm_fact) -> Tuple[Any, List[Union[int, Any]], List[Union[int, Any]], Any, Union[int, Any]]:
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
    copy_mask: np.ndarray
             Segmentation mask
    pred_mask: np.ndarray
             Mask with post-processing and with contours substracted
    """
    conv_factor, target_cnt, target_infos = mm_per_pixel(target_model, img_path, resize, norm_fact)
    # conv_factor, target_cnt = 1, []
    if len(target_cnt) == 0:
        conv_factor = 1
        raise(Warning('No targets were detected.'))

    # Mask and contour predictions
    pred_mask = full_prediction(mask_model, img_path, patch_size, resize, norm_fact=norm_fact)
    pred_contour = full_prediction(contours_model, img_path, patch_size, resize, 255)

    # Modal filter to eliminate artifacts at the junction of the predicted tiles
    pred_mask = modal(pred_mask, rectangle(7, 7))
    pred_contour = modal(pred_contour, rectangle(5, 5))
    copy_mask = pred_mask.copy()

    pred_mask = cv2.dilate(pred_mask, np.ones((3, 3), np.uint8), iterations=1)
    pred_mask = remove_small_objects(label(pred_mask), 1500)
    pred_mask[pred_mask != 0] = 255

    pred_contour = cv2.ximgproc.thinning(pred_contour)/255
    pred_contour = pred_contour.astype(np.uint8)

    pred_contour = remove_small_objects(label(pred_contour), 50)
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
    pred_mask = remove_small_objects(label(pred_mask), 1000)
    pred_mask[pred_mask != 0] = 255

    pred_mask = remove_small_holes(label(pred_mask), 3000)
    pred_mask[pred_mask != 0] = 255
    pred_mask = pred_mask.astype(np.uint8) * 255

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
            angle = ellipse[2]

            # Rotate shape
            angle = -angle if angle < 90 else 180 - angle
            rotated = rotate(np.squeeze(contour), (xc, yc), angle)
            x_sorted = sorted(rotated, key=lambda x: x[0])
            y_sorted = sorted(rotated, key=lambda x: x[1])
            WIDTH = x_sorted[-1][0] - x_sorted[0][0]
            HEIGHT = y_sorted[-1][1] - y_sorted[0][1]

            # Find closest target
            sorted_targets = sorted(target_infos, key=lambda x: np.sqrt((x['pos'][0] - xc)**2 + (x['pos'][1] - yc)**2))
            conv_factor = 40/sorted_targets[0]['dim']

            # Semi-axis (must be integers)
            widthE, heightE = ellipse[1]

            # Ellipse params
            a, b = round(0.5 * heightE) + 4, round(0.5 * widthE) + 4

            width_px = widthE + 8 if widthE < WIDTH else WIDTH + 8
            height_px = heightE + 8 if heightE < HEIGHT else HEIGHT + 8

            width_mm, height_mm = width_px * conv_factor, height_px * conv_factor

            # we filter the ellipses to eliminate those who are too small
            if 25 < width_mm < 150:
                cv2.ellipse(color_img, (xc, yc), (b, a), ellipse[2], 0, 360, (255, 255, 0), 2)
                rect = (ellipse[0], (width_px, height_px), ellipse[2])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(color_img, [box], 0, (0, 255, 255), 2)

                cv2.putText(color_img, f'{width_mm:.0f}x{height_mm:.0f}', (xc - 40, yc),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                # cv2.putText(color_img, f'{(WIDTH + 8) * conv_factor:.0f}x{(HEIGHT + 8) * conv_factor:.0f}', (xc - 25, yc + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                widths.append(width_mm)
                heights.append(height_mm)

            for box in target_cnt:
                # Draw contour of the target
                cv2.drawContours(color_img, [box], 0, (0, 255, 0), 3)

            # cv2.imwrite('preds.png', color_img)

    return color_img, widths, heights, copy_mask, pred_mask
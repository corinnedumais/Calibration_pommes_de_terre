import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects

from Utils.segmentation import full_prediction


def identify_targets(model, img_path):
    """
    Utility function to display target localization with red dots and green contour.
    """
    # Load the image in color
    color_img = Image.open(img_path)
    color_img = color_img.resize((2048, 1536), Image.ANTIALIAS)
    color_img = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)

    # Get target prediction
    pred = full_prediction(model, img_path, patch_size=256, resize=(2048, 1536), norm_fact=255)
    pred = modal(pred, rectangle(5, 5))

    plt.imshow(pred)
    plt.show()

    pred = remove_small_objects(label(pred), 1500)
    pred[pred != 0] = 255
    pred = pred.astype(np.uint8)

    # plt.imshow(pred)
    # plt.axis('off')
    # plt.show()

    # Get contours
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes = []

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
            # Draw contour of the target
            cv2.drawContours(color_img, [box], 0, (0, 255, 0), 5)

            # Draw middle point
            cv2.circle(color_img, np.int0(rect[0]), radius=5, color=(0, 0, 255), thickness=-1)

            half_diag = np.sqrt(2 * (min_dim / 2) ** 2)
            # Draw mid-side points
            cv2.circle(color_img, np.int0((rect[0][0] + min_dim / 2 * np.cos(np.deg2rad(rect[2])),
                                           rect[0][1] + min_dim / 2 * np.sin(np.deg2rad(rect[2])))), radius=5,
                       color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, np.int0((rect[0][0] - min_dim / 2 * np.cos(np.deg2rad(rect[2])),
                                           rect[0][1] - min_dim / 2 * np.sin(np.deg2rad(rect[2])))), radius=5,
                       color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, np.int0((rect[0][0] - min_dim / 2 * np.cos(np.deg2rad(90 - rect[2])),
                                           rect[0][1] + min_dim / 2 * np.sin(np.deg2rad(90 - rect[2])))), radius=5,
                       color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, np.int0((rect[0][0] + min_dim / 2 * np.cos(np.deg2rad(90 - rect[2])),
                                           rect[0][1] - min_dim / 2 * np.sin(np.deg2rad(90 - rect[2])))), radius=5,
                       color=(0, 0, 255), thickness=-1)

            # Draw corner points
            corner1 = np.int0((rect[0][0] + half_diag * np.cos(np.deg2rad(rect[2] + 45)),
                               rect[0][1] + half_diag * np.sin(np.deg2rad(rect[2] + 45))))
            corner3 = np.int0((rect[0][0] - half_diag * np.cos(np.deg2rad(rect[2] + 45)),
                               rect[0][1] - half_diag * np.sin(np.deg2rad(rect[2] + 45))))
            corner2 = np.int0((rect[0][0] - half_diag * np.cos(np.deg2rad(45 - rect[2])),
                               rect[0][1] + half_diag * np.sin(np.deg2rad(45 - rect[2]))))
            corner4 = np.int0((rect[0][0] + half_diag * np.cos(np.deg2rad(45 - rect[2])),
                               rect[0][1] - half_diag * np.sin(np.deg2rad(45 - rect[2]))))
            cv2.circle(color_img, corner1, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, corner3, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, corner2, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, corner4, radius=4, color=(0, 0, 255), thickness=-1)

    return color_img

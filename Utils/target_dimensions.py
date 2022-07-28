import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import splprep, splev
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects
from tensorflow import keras

from Models.Model import dice_loss, dice_coeff
from Utils.segmentation import full_prediction
from Utils.utils import show


def identify_targets(model, img_path):
    # Load the image in color
    color_img = Image.open(img_path)
    color_img = color_img.resize((2048, 1536), Image.ANTIALIAS)
    color_img = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)

    # Get target prediction
    pred = full_prediction(model, img_path, patch_size=256, resize=(2048, 1536), norm_fact=255)
    pred = modal(pred, rectangle(5, 5))

    pred = remove_small_objects(label(pred), 1500)
    pred[pred != 0] = 255
    pred = pred.astype(np.uint8)

    # plt.imshow(pred)
    # plt.axis('off')
    # plt.show()

    # Get contours
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes = []

    deltax, deltay = [], []
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
            cv2.drawContours(color_img, [box], 0, (0, 255, 0), 3)

            # Draw middle point
            cv2.circle(color_img, np.int0(rect[0]), radius=5, color=(0, 0, 255), thickness=-1)

            half_diag = np.sqrt(2*(min_dim/2)**2)
            # Draw mid-side points
            cv2.circle(color_img, np.int0((rect[0][0] + min_dim/2*np.cos(np.deg2rad(rect[2])), rect[0][1] + min_dim/2*np.sin(np.deg2rad(rect[2])))), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, np.int0((rect[0][0] - min_dim/2*np.cos(np.deg2rad(rect[2])), rect[0][1] - min_dim/2*np.sin(np.deg2rad(rect[2])))), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, np.int0((rect[0][0] - min_dim/2*np.cos(np.deg2rad(90-rect[2])), rect[0][1] + min_dim/2*np.sin(np.deg2rad(90-rect[2])))), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, np.int0((rect[0][0] + min_dim/2*np.cos(np.deg2rad(90-rect[2])), rect[0][1] - min_dim/2*np.sin(np.deg2rad(90-rect[2])))), radius=5, color=(0, 0, 255), thickness=-1)

            # Draw corner points
            corner1 = np.int0((rect[0][0] + half_diag*np.cos(np.deg2rad(rect[2]+45)), rect[0][1] + half_diag*np.sin(np.deg2rad(rect[2]+45))))
            corner3 = np.int0((rect[0][0] - half_diag*np.cos(np.deg2rad(rect[2]+45)), rect[0][1] - half_diag*np.sin(np.deg2rad(rect[2]+45))))
            corner2 = np.int0((rect[0][0] - half_diag*np.cos(np.deg2rad(45-rect[2])), rect[0][1] + half_diag*np.sin(np.deg2rad(45-rect[2]))))
            corner4 = np.int0((rect[0][0] + half_diag*np.cos(np.deg2rad(45-rect[2])), rect[0][1] - half_diag*np.sin(np.deg2rad(45-rect[2]))))
            cv2.circle(color_img, corner1, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, corner3, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, corner2, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(color_img, corner4, radius=4, color=(0, 0, 255), thickness=-1)

            deltax.extend([abs(corner2[0] - corner1[0]), abs(corner3[0] - corner2[0]), abs(corner4[0] - corner3[0]), abs(corner1[0] - corner4[0])])
            deltay.extend([abs(corner2[1] - corner1[1]), abs(corner3[1] - corner2[1]), abs(corner4[1] - corner3[1]), abs(corner1[1] - corner4[1])])

    deltax = [i for i in deltax if i > 15]
    deltay = [i for i in deltay if i > 15]

    m = []
    for dx, dy in zip(deltax, deltay):
        theta = np.arctan(dy/dx)
        m.append(40 * np.cos(theta)/dx)
        m.append(40 * np.sin(theta)/dx)

    m = np.mean(m)

    # Mean size of all detected targets
    mean_size = np.mean(sizes)
    mm_per_pixel = 40 / mean_size

    return color_img, mm_per_pixel


model = keras.models.load_model('Trained Models/targets.h5',
                                custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

for i in range(1, 2):
    file_name = f'test14.jpg'
    target_im, conversion_factor = identify_targets(model, f'PDT detection/SolanumTuberosum/Test_images/{file_name}')

    # print(conversion_factor)
    # cv2.putText(target_im, f'Facteur de conversion: {conversion_factor:.4} mm/px', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
    #             (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.imwrite(f'res{file_name}', target_im)
    show(target_im, dims=(800, 600))

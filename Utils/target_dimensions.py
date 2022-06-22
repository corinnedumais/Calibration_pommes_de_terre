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
    pred = full_prediction(model, img_path, patch_size=256, resize=(2048, 1536))
    pred = modal(pred, rectangle(5, 5))

    pred = remove_small_objects(label(pred), 1500)
    pred[pred != 0] = 255
    pred = pred.astype(np.uint8)

    plt.imshow(pred)
    plt.axis('off')
    plt.show()

    # Get contours
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes = []
    for contour in contours:
        # Fit rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1][0], rect[1][1]
        # Check if ressembles enough a square
        if 0.75 < width / height < 1.25:
            sizes.append((width + height) / 2)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Draw contour of the target
            cv2.drawContours(color_img, [box], 0, (0, 255, 0), 2)

    # Mean size of all detected targets
    mean_size = np.mean(sizes)
    mm_per_pixel = 40 / mean_size

    return color_img, mm_per_pixel


model = keras.models.load_model('Trained Models/targets_8c_no_norm.h5',
                                custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

for i in range(1, 9):
    file_name = f'{i:02}.jpg'
    s_time = time.time()
    target_im, conversion_factor = identify_targets(model, f'Target detection/Dataset Target/Eval/Images/{file_name}')
    print(time.time() - s_time)

    # print(conversion_factor)
    # cv2.putText(target_im, f'Facteur de conversion: {conversion_factor:.4} mm/px', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
    #             (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.imwrite(f'res{file_name}', target_im)
    show(target_im)

import cv2
import numpy as np
from PIL import Image
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects

from Utils.segmentation import full_prediction


class Calibrator:
    def __init__(self, mask_model, contour_model, target_model, weight_model):
        self._mm = mask_model
        self._cm = contour_model
        self._tm = target_model
        self._wm = weight_model

    def calibrate(self, img_path):
        pass

    def _mm_per_pixel(self, img_path):
        # Get target prediction
        pred = full_prediction(self._tm, img_path=img_path, patch_size=256, resize=(2048, 1536))
        pred = modal(pred, rectangle(5, 5))
        pred = remove_small_objects(label(pred), 1500)
        pred[pred != 0] = 255
        pred = pred.astype(np.uint8)

        # Get contours
        contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sizes = []
        for contour in contours:
            # Fit rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1][0], rect[1][1]
            # Check if ressembles enough a square
            if 0.85 < width / height < 1.15:
                sizes.append((width + height) / 2)

        # Mean size of all detected targets
        mean_size = np.mean(sizes)
        mm_per_pixel = 40 / mean_size
        return mm_per_pixel

    @staticmethod
    def _resize(img, size):
        if isinstance(img, str):
            img = Image.open(img)
            img = img.resize(size, Image.ANTIALIAS)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            img = img.resize(size, Image.ANTIALIAS)
        return np.array(img)

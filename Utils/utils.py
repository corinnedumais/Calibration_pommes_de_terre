# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage.filters.rank import modal
from skimage.measure import label
from skimage.morphology import rectangle, remove_small_objects

from Utils.segmentation import full_prediction


def normalize_dataset(dataset: np.ndarray) -> np.ndarray:
    """
    Function to normalize each attribut of a dataset. Minimum value will be 0 and maximum value will be 1.

    Parameters
    ----------
    dataset: np.ndarray
             Dataset to normalize. Should be shape (n, m) where n is number of samples and m is the number of features.

    Returns
    -------
    np.ndarray
               Dataset with each attribut individually normalized.
    """
    # Find the min and max values for each column
    min_max = []

    for i in range(len(dataset[0])):
        col_values = dataset.T[i]
        min_max.append((min(col_values), max(col_values)))

    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

    return dataset


def show(img, dims=(1536, 2048)):
    """
    Utility function to facilitate display of result with cv2.

    Parameters
    ----------
    img
        Image to display.
    dims: tuple of int
          Display dimensions (default is 1536 x 2048)
    """
    cv2.namedWindow('PDT_detection', cv2.WINDOW_NORMAL)
    im = cv2.resizeWindow('PDT_detection', dims[0], dims[1])
    cv2.imshow('PDT_detection', img)
    cv2.waitKey(0)

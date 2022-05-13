# -*- coding: utf-8 -*-
import numpy as np


def pixels_to_mm(px_measure: float, original_resolution: int, resize_factor: float) -> float:
    """
    Function to convert a the length of pixel segment to millimeters (mm).

    Since 1 inch = 25.4 mm, conversion is done using resolution in ppi, with a correction to account for resizing.

    Parameters
    ----------
    px_measure: float
                Measure to convert to mm.
    original_resolution: int
                Resolution (horizontal and vertical) of original image in ppi (pixels per inches)
    resize_factor: float
                Factor by which the image was resized.

    Returns
    -------
    float
          Measure in millimeters.
    """
    return px_measure * 25.4/original_resolution * resize_factor


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

# -*- coding: utf-8 -*-

import numpy as np


def pixels_to_mm(px_measure, angle, original_resolution, resize_factor):
    """
    Function to convert a the length of pixel segment to mm.

    Since 1po = 25.4mm, conversion is done using resolution in ppi. Because the resolution is horizontal/vertical,
    a correction is added to account for the orientation of the segment.

    Parameters:
        px_measure (int or float): Measure to convert to mm
        angle (float): Angle of the segment in degrees. Should be between 0 and 180, measured as
        0
        |
        ---- 90
        |
        180
        original_resolution (int): Resolution (horizontal and vertical) of original image in ppi (pixels per inches)
        resize_factor (int or float): Factor by which the image was resized
    """
    if 0 <= angle <= 45:
        rot_ajust = np.cos(np.deg2rad(angle))
    elif 45 < angle <= 90:
        rot_ajust = np.cos(np.deg2rad(90 - angle))
    elif 90 < angle <= 135:
        rot_ajust = np.cos(np.deg2rad(abs(90 - angle)))
    else:
        rot_ajust = abs(np.cos(np.deg2rad(angle)))

    if resize_factor < 0:
        resize_factor = 1/resize_factor

    rot_ajust = 1

    mm_measure = px_measure * 25.4/original_resolution * resize_factor * 1/rot_ajust
    return mm_measure


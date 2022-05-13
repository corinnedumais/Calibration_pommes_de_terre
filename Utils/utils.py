# -*- coding: utf-8 -*-


def pixels_to_mm(px_measure, original_resolution, resize_factor):
    """
    Function to convert a the length of pixel segment to mm.

    Since 1po = 25.4mm, conversion is done using resolution in ppi, with a correction to account for resizing.

    Parameters:
        px_measure (int or float): Measure to convert to mm
        original_resolution (int): Resolution (horizontal and vertical) of original image in ppi (pixels per inches)
        resize_factor (int or float): Factor by which the image was resized
    """
    return px_measure * 25.4/original_resolution * resize_factor


def normalize_dataset(dataset):
    # Find the min and max values for each column
    min_max = []
    for i in range(len(dataset[0])):
        col_values = dataset.T[i]
        min_max.append((min(col_values), max(col_values)))

    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
    return dataset

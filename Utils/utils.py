# -*- coding: utf-8 -*-
import numbers
from collections import OrderedDict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress


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


def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, **kwargs):
    """
    Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the q-q plot
    rmse = np.linalg.norm(y_quantiles - x_quantiles) / np.sqrt(len(x_quantiles))
    mean_bias = np.mean(np.subtract(y_quantiles, x_quantiles))
    mae = np.mean(np.abs(np.subtract(y_quantiles, x_quantiles)))
    ax.errorbar(x_quantiles, y_quantiles, xerr=1, fmt='o', label=f'RMSE:{rmse:.2f}\n'
                                                                 f'MAE:{mae:.2f}\n'
                                                                 f'MBE:{mean_bias:.2f}', **kwargs)
    return x_quantiles, y_quantiles


def show_QQplot_width_length(d, h, real_d, real_h, variety):
    """
    Function to format and display QQ plots for both width and length.

    X-axis is for reference or 'real' quantiles. Y-axis is for predicted quantiles.

    Parameters
    ----------
    d: array-like
        Predicted diameters
    h: array-like
        Predicted heights
    real_d:
        Reference diameters
    real_h:
        Reference heights
    variety: str
        variety of potato, should be 'burbank' or 'mountain_gem'
    """

    assert variety in ['burbank', 'mountain_gem'], 'The variety name given is not valid.'
    assert len(d) == len(h), 'Arrays for predicted widths and diameters should be of same length.'
    assert len(real_d) == len(real_h), 'Arrays for reference widths and diameters should be of same length.'

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 7))

    # Plot the width QQ plot, its linear regression and evaluation metrics
    x, y = qqplot(real_d, d, ax=ax1, color='#DB6666', mec='k')
    res = linregress(x, y)
    ax1.plot(x, [res.slope * i + res.intercept for i in x], 'tab:blue', label=f'y=ax+b\n'
                                                                              f'a={res.slope:.2f}$\pm${res.stderr:.2f}\n'
                                                                              f'b={res.intercept:.0f}$\pm${res.intercept_stderr:.0f}\n'
                                                                              f'R$^2$={res.rvalue ** 2:.2f}')
    ax1.set_xlabel('Quantiles théoriques [mm]', fontsize=16)
    ax1.set_ylabel('Quantiles prédits [mm]', fontsize=16)

    if variety == 'mountain_gem':
        variety = 'Mountain Gem'
    else:
        variety = 'Burbank'

    ax1.set_title(f'Q-Q plot pour la largeur ({variety})', fontsize=18)
    ax1.legend(loc=4, fontsize=16)

    # Plot the length QQ plot, its linear regression and evaluation metrics
    x2, y2 = qqplot(real_h, h, ax=ax2, color='#DB6666', mec='k')
    res2 = linregress(x2, y2)
    ax2.plot(x2, [res2.slope * i + res2.intercept for i in x2], 'tab:blue', label=f'y=ax+b\n'
                                                                                  f'a={res2.slope:.2f}$\pm${res2.stderr:.2f}\n'
                                                                                  f'b={res2.intercept:.0f}$\pm${res2.intercept_stderr:.0f}\n'
                                                                                  f'R$^2$={res2.rvalue ** 2:.2f}')
    ax2.set_xlabel('Quantiles théoriques [mm]', fontsize=16)
    ax2.set_ylabel('Quantiles prédits [mm]', fontsize=16)
    ax2.set_title(f'Q-Q plot pour la longueur ({variety})', fontsize=18)
    ax2.legend(loc=4, fontsize=16)

    plt.tight_layout()
    plt.show()


def get_calibres(diameters):
    """
    Function to get the caliber of a sample from its diameters.

    Parameters
    ----------
    diameters: array-like
        Diameters of all elements of the sample in millimeters.

    Returns
    -------
        OrderedDict: calibers
            Ordered dictionary with each caliber class and the number of objects in each.
    """

    calibers = OrderedDict([('a) 3 po+', 0), ('b) 3 po', 0), ('c) 2 1/2 po', 0), ('d) 2 1/4 po', 0),
                            ('e) 2 po', 0), ('f) 1 7/8 po', 0), ('g) 1 3/4 po', 0)])
    for mm in diameters:
        # Convert mm to po since caliber classes are named in inches
        inches = mm / 25.4
        if inches >= 3:
            calibers['a) 3 po+'] += 1
        elif inches >= 2.5:
            calibers['b) 3 po'] += 1
        elif inches >= 2.25:
            calibers['c) 2 1/2 po'] += 1
        elif inches >= 2:
            calibers['d) 2 1/4 po'] += 1
        elif inches >= 1.875:
            calibers['e) 2 po'] += 1
        elif inches >= 1.75:
            calibers['f) 1 7/8 po'] += 1
        else:
            calibers['g) 1 3/4 po'] += 1
    return calibers

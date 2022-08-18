from os.path import join

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def make_performance_plots(dice_train, dice_val, acc_train, acc_val, loss_train, loss_val):
    """
    Utility function to format and save performance plots.

    Plot includes three figures to display training and validation curves for loss, accuracy and dice coefficient.

    Parameters
    ----------
    dice_train: str
        Name of the text file
    dice_val: str
        Name of the text file
    acc_train: str
        Name of the text file
    acc_val: str
        Name of the text file
    loss_train: str
        Name of the text file
    loss_val: str
        Name of the text file

    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax = axes.ravel()

    ax[0].plot(np.loadtxt(loss_train, skiprows=1, usecols=(1,), delimiter=','),
               np.loadtxt(loss_train, skiprows=1, usecols=(2,), delimiter=','), label='Entraînement')
    ax[0].plot(np.loadtxt(loss_val, skiprows=1, usecols=1, delimiter=','),
               np.loadtxt(loss_val, skiprows=1, usecols=2, delimiter=','), '--', label='Évaluation')
    ax[0].set_xlabel('Epoch', fontsize=13)
    ax[0].set_ylabel('Fonction de coût', fontsize=13)
    ax[0].legend(fontsize=13)

    ax[1].plot(np.loadtxt(acc_train, skiprows=1, usecols=1, delimiter=','),
               np.loadtxt(acc_train, skiprows=1, usecols=2, delimiter=','), label='Entraînement')
    ax[1].plot(np.loadtxt(acc_val, skiprows=1, usecols=1, delimiter=','),
               np.loadtxt(acc_val, skiprows=1, usecols=2, delimiter=','), '--', label='Évaluation')
    ax[1].set_xlabel('Epoch', fontsize=13)
    ax[1].set_ylabel('Exactitude (accuracy)', fontsize=13)
    ax[1].legend(fontsize=13)

    ax[2].plot(np.loadtxt(dice_train, skiprows=1, usecols=1, delimiter=','),
               np.loadtxt(dice_train, skiprows=1, usecols=2, delimiter=','), label='Entraînement')
    ax[2].plot(np.loadtxt(dice_val, skiprows=1, usecols=1, delimiter=','),
               np.loadtxt(dice_val, skiprows=1, usecols=2, delimiter=','), '--', label='Évaluation')
    ax[2].set_xlabel('Epoch', fontsize=13)
    ax[2].set_ylabel('Coefficient Dice', fontsize=13)
    ax[2].legend(fontsize=13)

    plt.tight_layout()
    fig.savefig('target_performance_plots.png')


dir = 'Courbes'
type = 'mask'
make_performance_plots(join(dir, f'{type}_dice_train.csv'),
                       join(dir, f'{type}_dice_val.csv'),
                       join(dir, f'{type}_acc_train.csv'),
                       join(dir, f'{type}_acc_val.csv'),
                       join(dir, f'{type}_loss_train.csv'),
                       join(dir, f'{type}_loss_val.csv'))

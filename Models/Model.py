# -*- coding: utf-8 -*-

from __future__ import division

import logging
import os
from contextlib import redirect_stdout

import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Lambda, add
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

# To avoid the display of certain warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class UNetST:
    """
    Class representing a U-Net neural network to segment potatoes
    """

    def __init__(self, input_size, output_classes, kernel_size=(3, 3), channels=16, batchnorm=True):
        """
        Parameters:
            input_size (tuple): Size of the input data (height, width, channels)
            output_classes (int): Number of classes to segment
            kernel_size (tuple): Size of the convolutional kernels (height, width)
            channels (int): Number of channels for the first convolutional layer.
        """
        # Network parameters
        self.input_size = input_size
        self.output_classes = output_classes
        self.kernel_size = kernel_size
        self.channels = channels
        self.batchnorm = batchnorm

        # Parameters for the convolutional layers
        self.init = 'he_normal'
        self.act = 'relu'

        # Set optimizer and loss function
        self.optimizer = Adam(learning_rate=1e-4)
        self.loss = binary_crossentropy

    def Conv2D_block(self, input_tensor, channels, kernel_size, kernel_initializer, padding='same'):
        """
        Method to add 2 convolutional layers
        """
        x = Conv2D(channels, kernel_size=kernel_size, kernel_initializer=kernel_initializer, padding=padding)(input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation(self.act)(x)

        x = Conv2D(filters=channels, kernel_size=kernel_size, kernel_initializer=kernel_initializer, padding=padding)(x)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation(self.act)(x)

        return x

    def dilated_bottleneck(self, input_db):
        dilate1 = Conv2D(8*self.channels, 3, activation='relu', padding='same', dilation_rate=1, kernel_initializer='he_normal')(input_db)
        b7 = BatchNormalization()(dilate1)
        dilate2 = Conv2D(8*self.channels, 3, activation='relu', padding='same', dilation_rate=2, kernel_initializer='he_normal')(b7)
        b8 = BatchNormalization()(dilate2)
        dilate3 = Conv2D(8*self.channels, 3, activation='relu', padding='same', dilation_rate=4, kernel_initializer='he_normal')(b8)
        b9 = BatchNormalization()(dilate3)
        dilate4 = Conv2D(8*self.channels, 3, activation='relu', padding='same', dilation_rate=8, kernel_initializer='he_normal')(b9)
        b10 = BatchNormalization()(dilate4)
        dilate5 = Conv2D(8*self.channels, 3, activation='relu', padding='same', dilation_rate=16, kernel_initializer='he_normal')(b10)
        b11 = BatchNormalization()(dilate5)
        dilate6 = Conv2D(8*self.channels, 3, activation='relu', padding='same', dilation_rate=32, kernel_initializer='he_normal')(b11)
        dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
        return dilate_all_added

    def build(self):
        """
        Method to build and compile the U-Net neural network.

        Returns:
            Model: A U-Net neural network model.
        """

        # Block 1 (input)
        inputs = Input(self.input_size)
        # inputs = Lambda(lambda x: x / 255)(inputs)
        conv1 = self.Conv2D_block(inputs, self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 2
        pool1 = MaxPooling2D()(conv1)
        conv2 = self.Conv2D_block(pool1, 2*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 3
        pool2 = MaxPooling2D()(conv2)
        conv3 = self.Conv2D_block(pool2, 4*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 4
        pool3 = MaxPooling2D()(conv3)
        conv4 = self.Conv2D_block(pool3, 8*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 5 (bottleneck)
        conv5 = self.dilated_bottleneck(conv4)
        # pool4 = MaxPooling2D()(conv4)
        # conv5 = self.Conv2D_block(pool4, 16*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 6
        up1 = UpSampling2D()(conv5)
        merge6 = concatenate([conv4, up1], axis=-1)
        conv6 = self.Conv2D_block(merge6, 8*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 7
        up2 = UpSampling2D()(conv6)
        merge7 = concatenate([conv3, up2], axis=-1)

        conv7 = self.Conv2D_block(merge7, 4*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 8
        up3 = UpSampling2D()(conv7)
        merge8 = concatenate([conv2, up3], axis=-1)
        conv8 = self.Conv2D_block(merge8, 2*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 9 (output)
        up4 = UpSampling2D()(conv8)
        merge9 = concatenate([conv1, up4], axis=-1)

        conv9 = self.Conv2D_block(merge9, self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)
        conv9 = Conv2D(self.output_classes, kernel_size=(1, 1), activation='sigmoid', padding='same')(conv9)

        model = Model(inputs=inputs, outputs=conv9)

        model.compile(optimizer=self.optimizer, loss=dice_loss, metrics=['accuracy', dice_coeff])

        return model

    def save_summary(self):
        """
        Method to save in a text file the summary of the model.
        """
        with open('modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                self.build().summary(line_length=145)


def dice_loss(y_true, y_pred):
    """
    Function to calculate model loss with dice coefficient.
    Parameters:
        y_true: Ground truth
        y_pred: Prediction
    Returns:
         float: Dice loss
    """
    dice = tf.reduce_mean(tf.math.multiply(y_true, y_pred)) / (tf.reduce_mean(y_true) + tf.reduce_mean(y_pred)) * 2
    return 1 - dice


def dice_coeff(y_true, y_pred):
    """
    Function to calculate dice coefficient,
    Parameters:
        y_true: Ground truth
        y_pred: Prediction
    Returns:
         float: Dice coefficient
    """
    return tf.reduce_mean(tf.math.multiply(y_true, y_pred)) / (tf.reduce_mean(y_true) + tf.reduce_mean(y_pred)) * 2


def weighted_bce(y_true, y_pred):
    """
    Function to calculate weighted binary crossentropy loss.
    Parameters:
        y_true: Ground truth
        y_pred: Prediction
    Returns:
         float: Weighted binary crossentropy loss
    """
    weights = (y_true * 90.) + 1.
    # From logits = False puisqu'on a appliqué une sigmoide
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    w_bce = tf.reduce_mean(tf.expand_dims(bce, axis=-1) * weights)
    return w_bce


def combined(y_true, y_pred):
    return weighted_bce(y_true, y_pred) + dice_loss(y_true, y_pred)
import keras
import tensorflow as tf
import numpy as np

from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

# load the model
from Models.Model import dice_loss, dice_coeff

model = keras.models.load_model('Trained Models/mask_yellow.h5',
                                custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

# redefine model to output right after the first hidden layer
ixs = [12, 19, 26, 33]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

# load image
img = tf.keras.utils.load_img('PDT detection\SolanumTuberosum\TrainImages\img_0187.png')
img = np.expand_dims(np.array(img), axis=0)

# get feature map for first hidden layer
feature_maps = model.predict(img)

# plot the output from each block
square = 4
for fmap in feature_maps:
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(fmap[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()

from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# img = imread('PDT detection/SolanumTuberosum/TrainImages/img_0993.png')
# mean_pool=block_reduce(img, block_size=(9,9,1), func=np.mean)
# max_pool=block_reduce(img, block_size=(9,9,1), func=np.max)
# min_pool=block_reduce(img, block_size=(9,9,1), func=np.min)
#
# plt.figure(1)
# plt.subplot(221)
# imgplot = plt.imshow(img)
# plt.title('Original Image')
#
# plt.subplot(222)
# imgplot3 = plt.imshow(mean_pool)
# plt.title('Average pooling')
#
# plt.subplot(223)
# imgplot1 = plt.imshow(max_pool)
# plt.title('Max pooling')
#
# plt.subplot(224)
# imgplot1 = plt.imshow(min_pool)
# plt.title('Min pooling')
#
# plt.show()
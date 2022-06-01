import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Target detection/Images/im22.jpg'

im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
color_im = cv2.imread(path)

edges = cv2.Canny(im,100,200)

plt.imshow(edges)
plt.show()

cnt, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(color_im, cnt, -1, (0, 255, 0), 3)

plt.imshow(color_im, cmap='gray')
plt.show()

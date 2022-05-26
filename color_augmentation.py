import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('SolanumTuberosum/Images/01.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_brown = np.array([0, 20, 0])
upper_brown = np.array([100, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_brown, upper_brown)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
res[np.where((res == [0, 0, 0]).all(axis=2))] = [255, 0, 0]

plt.figure(figsize=(8, 8))
plt.imshow(res, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# cv2.imwrite('01_black.jpg', res)

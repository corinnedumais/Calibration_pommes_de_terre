import glob
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# termination criteria
from PIL import Image

from Utils.utils import show

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = [i for i in os.listdir('Utils/calib_photos')]
for fname in images:
    print(fname)
    img = Image.open(os.path.join('Utils/calib_photos', fname))
    img = img.resize((2048, 1536), Image.ANTIALIAS)
    img = np.array(img)
    # img = cv.imread(os.path.join('Utils/calib_photos', fname))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(th, (9, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        # cv.drawChessboardCorners(img, (9, 6), corners2, ret)
        # show(img, dims=(800, 600))
# cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = Image.open('PDT detection/SolanumTuberosum/Test_images/test14.jpg')
img = img.resize((2048, 1536), Image.ANTIALIAS)
img = np.array(img)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

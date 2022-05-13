# -*- coding: utf-8 -*-

import cv2
import numpy as np


def show(img, dims=(1536, 2048)):
    cv2.namedWindow('PDT_detection', cv2.WINDOW_NORMAL)
    im = cv2.resizeWindow('PDT_detection', dims[0], dims[1])
    cv2.imshow('PDT_detection', img)
    cv2.waitKey(0)

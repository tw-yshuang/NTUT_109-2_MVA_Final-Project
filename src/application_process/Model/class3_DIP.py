import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from Model.BoundaryDescriptor import get_contours_binary, calc_contour_feature, draw_bbox


def get_canny(img, isShow=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 70, 150)
    result = np.hstack([gray, blurred, canny])
    return canny


def get_contours(img, isShow=True):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 92, 255, 0)
    num_white_extreme = len(np.where(imgray > 200)[0])
    num_black_extreme = len(np.where(imgray < 25)[0])
    if np.mean(imgray) < 127 or num_white_extreme < 50:
        thresh_white = 255 - thresh
    else:
        thresh_white = thresh
    _, contours, hierarchy = cv2.findContours(
        thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), -1)
    return img_contours


def get_c3_img(img, position_ls):
    for position in position_ls:
        x, y, w, h = position
        crop_img = img[y:y+h, x:x+w]
        img_contours = get_contours(crop_img, isShow=False)
        # cv2.imshow("Contours_img", img_contours)
        # cv2.waitKey(0)
        img[y:y+h, x:x+w] = img_contours  # 指定位置填充，大小要一样才能填充
        # cv2.imshow("Merge", img_raw)
    return img

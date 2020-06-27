import cv2 as cv
import numpy as np


def get_HitMiss(img, kernel):
    img = cv.morphologyEx(img, cv.MORPH_HITMISS, kernel)
    return img


def get_Dilation(img, kernel, iterations=None):
    img = cv.dilate(img, kernel, iterations)
    return img


def get_Erosion(img, kernel, iterations=None):
    img = cv.erode(img, kernel, iterations)
    return img


def get_open(img, kernel):
    img_open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_line = cv.subtract(img, img_open)
    return img_open, img_line


def resize_show(img, size, img_name, show_palce):
    '''
    resize show img 我是黃魚舜
    '''

    img = cv.resize(img, None, fx=size, fy=size,
                    interpolation=cv.INTER_NEAREST)
    cv.imshow(img_name, img)
    cv.moveWindow(img_name, show_palce[0], show_palce[1])

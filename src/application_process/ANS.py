import pandas as pd
# import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import sys
import os
from Model.BoundaryDescriptor import get_contours_binary, calc_contour_feature, get_crop_imgs, draw_bbox


def get_mask(img, en_pix):
    en_pix.split(' ')

    rle = list(map(int, en_pix.split(' ')))

    pixel, pixel_count = [], []
    [pixel.append(rle[i]) if i % 2 == 0 else pixel_count.append(rle[i])
        for i in range(0, len(rle)-1)]
    # print('pixel starting points:\n', pixel)
    # print('pixel counting:\n', pixel_count)

    rle_pixels = [list(range(pixel[i], pixel[i]+pixel_count[i]))
                  for i in range(0, len(pixel)-1)]
    # print('rle_pixels\n:', rle_pixels)

    rle_mask_pixels = sum(rle_pixels, [])
    # print('rle mask pixels:\n', rle_mask_pixels)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # plt.imshow(img)
    # plt.show()

    img_size = img.shape[0] * img.shape[1]
    mask_img = np.zeros((img_size, 1), dtype=int)
    mask_img[rle_mask_pixels] = 255
    l, b = img.shape[0], img.shape[1]
    mask = np.reshape(mask_img, (b, l)).T  # / 255

    return mask


def create_square_img(img, imgShape=None):
    img_w = img.shape[0]
    img_h = img.shape[1]
    if imgShape is None:
        imgShape = [0, 0]
        if img_h > img_w:
            imgShape[0] = img_h
            imgShape[1] = img_h
        else:
            imgShape[0] = img_w
            imgShape[1] = img_w

    # if img_w > img_h:
    square_img = np.zeros(
        (imgShape[0], imgShape[1], 3), dtype='uint8')
    center = imgShape[0] // 2
    h_half = img_h / 2
    w_half = img_w / 2
    if h_half == img_h // 2 and w_half == img_w // 2:
        h_half = int(h_half)
        w_half = int(w_half)
        square_img[center - w_half: center + w_half,
                   center - h_half: center + h_half] = img

    elif h_half == img_h // 2 and w_half != img_w // 2:
        h_half = int(h_half)
        w_half = img_w // 2
        square_img[center - w_half: center + w_half+1,
                   center - h_half: center + h_half] = img

    elif h_half != img_h // 2 and w_half == img_w // 2:
        h_half = img_h // 2
        w_half = int(w_half)
        square_img[center - w_half: center + w_half,
                   center - h_half: center + h_half+1] = img

    else:
        h_half = img_h // 2
        w_half = img_w // 2
        square_img[center - w_half: center + w_half+1,
                   center - h_half: center + h_half+1] = img

    return square_img
    # else:
    #     square_img = np.zeros(
    #         (img_h, img_h, 3), dtype='uint8')
    #     center = img_h // 2
    #     w_half = img_w / 2
    #     if w_half == img_w // 2:
    #         w_half = int(w_half)
    #         square_img[center - w_half: center + w_half, 0:img_h] = img
    #     else:
    #         w_half = img_w // 2
    #         square_img[center - w_half: center + w_half+1, 0:img_h] = img
    #     return square_img


def ans_img(img_path, isShow=False):
    df = pd.read_csv('doc/train.csv')
    imgs_path = 'Data/train_images'

    img_name = img_path.split('/')[-1]

    num_rows = []
    for i in range(df.shape[0]):
        if img_name == df.iloc[i, 0]:
            num_rows.append(i)

    total_contours_img = []
    for num_row in num_rows:
        folder_name = df.iloc[num_row, 1]
        img_path = '{}/{}/{}'.format(imgs_path, folder_name, img_name)
        en_pix = df.iloc[num_row, 2]

        img = cv2.imread(img_path)
        # print(en_pix)
        # print(len(en_pix.split(' ')))
        mask = get_mask(img, en_pix)
        mask = mask.astype('uint8')

        contours = get_contours_binary(mask)
        contours_img = cv2.drawContours(
            img.copy(), contours, -1, (0, 0, 255), -1)
        total_contours_img.append(contours_img)

        if isShow is True:
            cv2.namedWindow('A_{}'.format(folder_name), 0)
            cv2.resizeWindow('A_{}'.format(
                folder_name), (img.shape[1] // 1, img.shape[0] // 1))
            cv2.moveWindow('A_{}'.format(folder_name), 300, 400*folder_name)
            cv2.imshow('A_{}'.format(folder_name), contours_img)

    return total_contours_img


if __name__ == "__main__":
    a = 0

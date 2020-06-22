import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import os
from Model.BoundaryDescriptor import get_contours_binary, calc_contour_feature, draw_bbox


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


if __name__ == "__main__":
    df = pd.read_csv('doc/train.csv')

    imgs_path = 'Data/train_images/'

    for num_img in range(df.shape[0]):
        img_name = df.iloc[num_img, 0]
        folder_name = df.iloc[num_img, 1]
        img_path = '{}{}/{}'.format(imgs_path, folder_name, img_name)
        en_pix = df.iloc[num_img, 2]

        img = cv2.imread(img_path)
        # print(en_pix)
        # print(len(en_pix.split(' ')))
        mask = get_mask(img, en_pix)
        mask = mask.astype('uint8')

        contours = get_contours_binary(mask)
        # img_contours = cv2.drawContours(
        #     img.copy(), contours, -1, (255, 0, 0), 2)

        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # plt.imshow(mask)
        # plt.show()

        features = calc_contour_feature(img, contours, isShow=False)
        
        # 截圖？？？？
        img_bbox = draw_bbox(img, features, color=(
            0, 0, 255), width=1, isShow=True)

        if num_img / 50 == num_img // 50:
            print(num_img)
        # plt.show()

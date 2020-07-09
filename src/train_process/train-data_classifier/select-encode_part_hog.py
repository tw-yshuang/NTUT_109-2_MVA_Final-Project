import pandas as pd
# import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import sys
import os
from Model.BoundaryDescriptor import get_contours_binary, calc_contour_feature, get_crop_imgs


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


if __name__ == "__main__":
    df = pd.read_csv('doc/train.csv')

    imgs_path = 'Data/train_images'
    save_file_path = 'hog_data'

    for num_img in range(df.shape[0]-1000):
        img_name = df.iloc[num_img, 0]
        folder_name = df.iloc[num_img, 1]
        img_path = '{}/{}/{}'.format(imgs_path, folder_name, img_name)
        en_pix = df.iloc[num_img, 2]

        img = cv2.imread(img_path)
        # print(en_pix)
        # print(len(en_pix.split(' ')))
        mask = get_mask(img, en_pix)
        mask = mask.astype('uint8')

        contours = get_contours_binary(mask)

        # get features of contours and img
        raw_features = calc_contour_feature(img, contours)

        # Beacuse there are some shape of encod_pixel is too small (e.g. img's size: 16*1 ),
        # let final-projectp's model effieient get lower, so we need to remove those encod_pixel.
        features = []
        for raw_feature in raw_features:
            (x, y, w, h) = raw_feature[3]
            if w > 5 and h > 5:
                features.append(raw_feature)

        # get crop imgs, there are several imgs in the crop_imgs, crop_imgs type is list
        crop_imgs = get_crop_imgs(img, features, 3, 3)

        for i in range(len(crop_imgs)):
            folder_path = '{}/{}/{}'.format(imgs_path,
                                            save_file_path, folder_name)
            save_img_path = '{}/{}_{}.jpg'.format(
                folder_path, img_name[:-4], (i + 1))

            # resize_img = cv2.resize(
            #     crop_imgs[i], (crop_imgs[i].shape[1]*10, crop_imgs[i].shape[0]*10))
            # cv2.imshow("resize", resize_img)
            # cv2.waitKey(0)
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            #     break

            if os.path.exists(folder_path) is False:
                os.makedirs(folder_path)
            cv2.imwrite(save_img_path, crop_imgs[i])

        if num_img % 50 == 0:
            print(num_img)

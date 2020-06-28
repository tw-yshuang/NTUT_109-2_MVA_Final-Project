import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np


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


def get_contours_binary(imgray):
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(type(imgray))
    imgray = imgray.astype('uint8')
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh_white = thresh

    # if your python-cv version is less than 4.0 the cv2.findContours will return 3 variable
    _, contours, hierarchy = cv2.findContours(
        thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours


if __name__ == "__main__":
    df = pd.read_csv('doc/train.csv')

    imgs_path = 'data/train_images/'

    for num_img in range(df.shape[0]):
        img_name = df.iloc[num_img, 0]
        folder_name = df.iloc[num_img, 1]
        img_path = '{}{}/{}'.format(imgs_path, folder_name, img_name)
        en_pix = df.iloc[num_img, 2]

        img = cv2.imread(img_path)
        # print(en_pix)
        # print(len(en_pix.split(' ')))
        mask = get_mask(img, en_pix)

        contours = get_contours_binary(mask)
        img_contours = cv2.drawContours(
            img.copy(), contours, -1, (255, 0, 0), 2)

        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # plt.imshow(mask)
        # plt.show()

        plt.subplot(2, 1, 1)
        plt.imshow(img, cmap='gray', interpolation='bilinear')
        plt.title("oringal {}".format(
            img_name)), plt.xticks([]), plt.yticks([])

        plt.subplot(2, 1, 2)
        plt.imshow(img_contours, cmap='gray', interpolation='bilinear')
        plt.title("contours {}".format(
            img_name)), plt.xticks([]), plt.yticks([])

        plt.show()
        plt.savefig("{}_select-encode.png".format(img_path[:-4]))

        if num_img / 50 == num_img // 50:
            print(num_img)
        # plt.show()

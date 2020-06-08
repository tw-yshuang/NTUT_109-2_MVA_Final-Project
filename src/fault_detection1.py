import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv('doc/train.csv')

    imgs_path = 'data/train_images/'

    for num_img in range(df.shape[0]):
        img_name = df.iloc[num_img, 0]
        folder_name = df.iloc[num_img, 1]
        img_path = '{}{}/{}'.format(imgs_path, folder_name, img_name)

        en_pix = df.iloc[num_img, 2]

        # print(en_pix)

        # print(len(en_pix.split(' ')))
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

        img = cv2.imread(img_path)
        # cv2.imshow("img", img)
        # plt.imshow(img)
        # plt.show()
        img_size = img.shape[0] * img.shape[1]
        mask_img = np.zeros((img_size, 1), dtype=int)
        mask_img[rle_mask_pixels] = 255
        l, b = img.shape[0], img.shape[1]
        mask = np.reshape(mask_img, (b, l)).T

        plt.imshow(mask)
        plt.savefig("{}.png".format(img_path[:-4]))

        if num_img / 50 == num_img // 50:
            print(num_img)
        # plt.show()

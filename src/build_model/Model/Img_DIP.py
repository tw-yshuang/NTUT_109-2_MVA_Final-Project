import numpy as np
import cv2


class ImgDIP(object):
    '''
    Image use, this program can setting lower and upper bounds that image colors you want to get,
    the setting mode can be RGB, HSV
    '''

    def __init__(self, img=None, img_path=None):
        if img is None and img_path is not None:
            self.img = cv2.imread(img_path)
        elif img is None:
            print("Error, this class need to choise img or img_path to import")

        self.img = img
        self.img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def split_channels(self, img=None):
        if img is None:
            img = self.img

        C1, C2, C3 = cv2.split(img)
        return C1, C2, C3

    def merge_channels(self, C1, C2, C3):
        img = cv2.merge([C1, C2, C3])
        return img

    def increase_256_channel(self, channel, NUM_INCREASE):
        rows = channel.shape[0]
        cols = channel.shape[1]

        for row in range(rows):
            for col in range(cols):
                item = channel[row, col]
                item += NUM_INCREASE
                if item > 255:
                    item = 255
                channel[row, col] = item

        return channel

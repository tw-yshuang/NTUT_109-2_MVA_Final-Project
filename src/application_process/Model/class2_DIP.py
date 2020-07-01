import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from Model.BoundaryDescriptor import get_contours_binary, calc_contour_feature, draw_bbox


def erode(gray, kernel=None, iterations=1, isShow=True):
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    img_erode = cv2.erode(gray, kernel, iterations=1)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 200
    maxLineGap = 5
    for i in range(60, 65, 5):
        threshold = i
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold, minLineLength, maxLineGap)
        img_cpy = img_erode.copy()
    # draw all lines
        for h in lines:
            (x1, y1, x2, y2) = h[0]
            img_ht = cv2.line(img_cpy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imshow("Hough line: " + str(threshold), img_ht)
        # cv2.waitKey(0)
    return img_erode, img_ht


def get_contours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 89, 255, 0)
    thresh_white = 255 - thresh
    _, contours, hierarchy = cv2.findContours(
        thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), -1)
    return img_contours


def get_c2_img(img, position_ls):
    for position in position_ls:
        x, y, w, h = position
        crop_img = img[y:y+h, x:x+w]
        img_erode, img_ht = erode(crop_img, isShow=False)
        # img_erode_countour = get_contours(img_erode, isShow=True)
        # cv2.imshow("Erode_img", img_erode)
        bbox_img = get_contours(img_ht)

        img[y:y+h, x:x+w] = bbox_img  # 指定位置填充，大小要一样才能填充
        # cv2.imshow("Merge", img)
    return img


if __name__ == "__main__":
    df = pd.read_csv('doc/train.csv')
    img_path = r'data\train_images\2\0b1e5a7a5.jpg'
    img_raw = cv2.imread(img_path)
    en_pix = df.iloc[308, 2]
    img = cv2.imread(img_path)
    mask = get_mask(img, en_pix)
    mask = mask.astype('uint8')
    contours = get_contours_binary(mask)
    features = calc_contour_feature(img, contours, isShow=False)
    # print(mask.shape)

    for feature in features:
        position = feature[3]
        x, y, w, h = position
        crop_img = img[y:y+h, x:x+w]
        img_erode, img_ht = erode(crop_img, isShow=True)
        # img_erode_countour = get_contours(img_erode, isShow=True)
        # cv2.imshow("Erode_img", img_erode)

        img_raw[y:y+h, x:x+w] = img_ht  # 指定位置填充，大小要一样才能填充
        cv2.imshow("Merge", img_raw)
        cv2.waitKey(0)
        cv2.destroyAllWindows

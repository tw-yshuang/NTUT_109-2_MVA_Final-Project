import numpy as np
import cv2


def window_size(ksize):
    kernel = np.ones((ksize, ksize), np.uint8)
    return kernel


def dilate(img, kernel, iterations=1, isShow=True):
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    return img_dilate


def erode(img, kernel, iterations=1, isShow=True):
    img_erode = cv2.erode(img, kernel, iterations=1)
    return img_erode


def convert_hsv(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    #(H, S, V) = split_color(image, False)
    return hsv_img, H, S, V


def image_process(image):
    hsv_img, H, S, V = convert_hsv(image)
    VS = np.uint8(np.clip(1 * V + 40, 0, 255))
    k_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    print("kernel", kernel)
    erosion = erode(VS, kernel)
    kernel_1 = window_size(5)
    dilation = dilate(erosion, kernel_1)
    erosion_2 = erode(dilation, kernel_1)
    return erosion_2


def get_contours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 92, 255, 0)
    thresh_white = 255 - thresh
    _, contours, hierarchy = cv2.findContours(
        thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), -1)
    return img_contours


def get_c1_img(img, position_ls):
    for position in position_ls:
        x, y, w, h = position
        crop_img = img[y: y + h, x: x + w]
        crop_img = image_process(crop_img)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
        bbox_img = get_contours(crop_img)
        img[y: y + h, x: x + w] = bbox_img
    return img


if __name__ == "__main__":
    image = cv2.imread(
        r"C:\Users\happy\Desktop\MVA\Final Project\1\02b2d3aa4.jpg")
    class_1_result = image_process(image)
    cv2.imshow('result', class_1_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

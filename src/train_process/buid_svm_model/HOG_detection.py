import cv2
import numpy as np
from Model.find_file_name import get_filenames


class Hog_Param(object):
    winSize = (128, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    def __init__(self, winSize=(64, 128), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8), nbins=9):
        # super(cv2).__init__()
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize

        self.nbins = nbins
        self.featureNum = int(
            ((self.winSize[0] - self.blockSize[0]) / self.blockStride[0] + 1) *
            ((self.winSize[1] - self.blockSize[1]) / self.blockStride[1] + 1) *
            ((self.blockSize[0] * self.blockSize[1]) / (self.blockStride[0] * self.blockStride[1])) * self.nbins)

        # self.hog = cv2.HOGDescriptor(
        #     self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins)

        return

    def DataLoader(self, posPaths, negPaths, img_w, img_h):
        self.posNum = len(posPaths)
        self.negNum = len(negPaths)
        self.featureArray = np.zeros(
            ((self.posNum + self.negNum), self.featureNum), np.float32)
        self.labelArray = np.zeros(((self.posNum + self.negNum), 1), np.int32)

        for i in range(self.posNum):
            self.get_imgHog_Hist(posPaths[i])
            self.labelArray[i, 0] = 1

        for i in range(self.negNum):
            self.get_imgHog_Hist(negPaths[i])
            self.labelArray[self.posNum+i, 0] = -1

    def get_imgHog_Hist(self, path):
        img = cv2.imread(self.winSize)
        img = cv2.resize(self.winSize)
        hist = steel.hog.compute(img, (8, 8))
        for i in range(self.featureNum):
            self.featureArray[i, j] = hist[j]
        return

    def build_svm(self, rate=0.01):
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.setC(rate)
        return

    def svm_train(self):
        # train
        ret = self.svm.train(
            self.featureArray, cv2.ml.ROW_SAMPLE, self.labelArray)
        # check
        alpha = np.zeros((1), np.float32)
        rho = self.svm.getDecisionFunction(0, alpha)
        print(rho)
        print(alpha)
        alphaArray = np.zeros((1, 1), np.float32)
        supportVArray = np.zeros((1, self.featureNum), np.float32)
        # resultArray = np.zeros((1, self.featureNum), np.float32)
        alphaArray[0, 0] = alpha
        resultArray = -1 * alphaArray * supportVArray
        # detect
        self.svm_detect = np.zeros((self.featureNum+1), np.float32)
        for i in range(self.featureNum):
            self.svm_detect[i] = resultArray[0, i]
        self.svm_detect[self.featureNum] = rho[0]

    def build_hog(self, svmDetect=None):
        if svmDetect is None:
            svmDetect = self.svm_detect
        svmDetect = self.svm_detect
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(svmDetect)
        return


def cv2_winClose():
    '''
    Press "q" to close all cv2 windows
    '''
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        return


if __name__ == "__main__":
    path = "Data/train_images/hog_data"
    file_extension = "jpg"
    filenames = get_filenames(path, file_extension)

    steel = Hog_Param()

    # positive data
    for i in range(len(filenames)):
        img = cv2.imread(filenames[i])
        cv2.imshow('a', img)
        cv2_winClose()
        img = cv2.resize((128, 64))
        hist = steel.hog.compute(img, (8, 8))
        for j in range(steel.featureNum):
            steel.featureArray[i, j] = hist[j]

        steel.labelArray[i, 0] = 1

import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, classification_report


def train_cv_svm(hog_sample, label, model_name="./Model/svm.xml", split=0.8):
    partition = int(len(hog_sample) * split)
    x_train, x_test = hog_sample[:partition], hog_sample[partition:]
    y_train, y_test = label[:partition].ravel(), label[partition:].ravel()
    # svm = cv2.ml.SVM() # old syntax
    SVM = cv2.ml.SVM_create()
    SVM.setType(cv2.ml.SVM_C_SVC)
    SVM.setKernel(cv2.ml.SVM_LINEAR)  # must us linear ##
    cv2.ml.SVM_LINEAR
    SVM.setC(0.01)
    #SVM.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 500, 1.e-06))
    SVM.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
    print("Training is done")
    predicted_label = SVM.predict(x_test)[1]
    print("Accuracy: "+str(accuracy_score(y_test, predicted_label)))
    print('\n')
    print(classification_report(y_test, predicted_label))
    print(len(SVM.getSupportVectors()))
    sv = SVM.getUncompressedSupportVectors()
    # I guess detector store the Linear Indicator
    SVM.save(model_name)
    tree = ET.parse(model_name)
    print("SVM Model is save as: ", model_name)
    return SVM


def load_detector(SVM):
    modelname = './model/svm_128_64.xml'
    SVM = cv2.ml.SVM_load(modelname)
    winSize = (128, 64)  # (128, 64) #(x, y) (64, 128) #(64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    HOG = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    svmvec1 = SVM.getSupportVectors()[0]  # combined sv
    rho = -SVM.getDecisionFunction(0)[0]
    svmvec = np.append(svmvec1, rho)  # 加入 W0
    detector = HOG.setSVMDetector(svmvec)
    return detector, HOG


def create_Hog_data(HOG, path="./Positive_Images",  isShow=True, class_start_index=1, isFlip=True):
    # class_start_index: class 起始編號
    # winSize = winSize  #(128, 64) #(x, y) (64, 128) #(64, 128)
    # blockSize = (16,16)
    # blockStride = (8,8)
    # cellSize = (8,8)
    # nbins = 9
    # #print(file_list)
    # HOG = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    data_list = list()
    label_list = list()
    # find dir first
    dir_list = findAllDir(path=path)
    for i, dirname in enumerate(dir_list):
        file_list = findAllImagFiles(dirname)
        for f in file_list:
            imgBGR = cv2.imread(f, 1)  # read gray image instead imgBGR
            img = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
            # print(img.shape)
            # (winSize[1], winSize[0])) if skimage
            img = cv2.resize(img, HOG.winSize)
            # skimage 用 (row, col), cv2 用 (x, y)
            hog_feature = HOG.compute(img)
            hog_feature = hog_feature.ravel()  # flatten all data
            #f_list = []
            # for f in hog_feature:
            #    f_list.append(f[0])
            # for display only
            # fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
            #            cells_per_block=(2, 2), visualize=True, multichannel=False)
            # print(fd)
            if isShow:
                cv2.imshow("Fish image", imgBGR)
                cv2.imshow("Fish hog image", hog_image)
                cv2.waitKey(0)
            data_list.append(hog_feature)
            label_list.append(i + class_start_index)
            if isFlip:
                img = cv2.flip(img, 1)  # flip horizontal
                # skimage 用 (row, col), cv2 用 (x, y)
                hog_feature = HOG.compute(img)
                hog_feature = hog_feature.ravel()  # flatten all data
                data_list.append(hog_feature)
                label_list.append(i + class_start_index)
    return data_list, label_list


def load_data(HOG, positve_path="./Positive_Images", positive_class_start_index=1, negative_path="./Negative_Images", negative_class_start_index=2):
    # generate_positive_image()
    #positive_list, label_list = create_Hog_data(path = "./Positive_Images" )
    #generate_negative_image(path = "./Images", label = 2 , isShow = False)
    t_start = time.perf_counter()
    positive_list, positive_label_list = create_Hog_data(
        HOG, path=positve_path,  class_start_index=positive_class_start_index, isShow=False)
    t_end = time.perf_counter()
    print("Create postive samples spent: ", (t_end - t_start), " sec.")
    t_start = time.perf_counter()
    negative_list, negative_label_list = create_Hog_data(
        HOG, path=negative_path, class_start_index=negative_class_start_index, isShow=False, isFlip=False)
    t_end = time.perf_counter()
    print("Create negative samples spent: ", (t_end - t_start), " sec.")
    # print(positive_list)
    # print(negative_list)
    sample = list()
    label = list()
    sample = positive_list + negative_list
    label = positive_label_list + negative_label_list
    sample = np.float32(sample)
    label = np.array(label)
    # Shuffle Samples
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(sample))
    sample = sample[shuffle]
    label = label[shuffle]
    print(sample)
    print(label)
    return sample, label


if __name__ == "__main__":
    # crate data, and set WinSize (64, 32) = (Width, Height)
    hog_sample, label = load_data(positve_path="./Positive_Images", positive_class_start_index=-1,
                                  negative_path="./Negative_Images", negative_class_start_index=1)
    # # #https://www.kaggle.com/manikg/training-svm-classifier-with-hog-features
    # # # use CV2.SVM instead: must use linearSVC
    SVM = train_cv_svm(hog_sample, label, model_name="./Model/svm_128_64.xml")

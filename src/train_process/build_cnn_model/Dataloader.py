import cv2
import numpy as np
import torch
from Model.find_file_name import get_filenames
from torch.utils.data import Dataset, DataLoader
from pre_process import dip_pre_process


class ImgDataset(Dataset):

    def __init__(self, filenames, classifier, imgResize=(64, 64), isTrain=False, isOneHotEncod=False, dataAutoBalance=True, rateMagnifyData=1.0):
        (img_h, img_w) = imgResize  # img resize width

        (datas, labels) = organize_dataset(
            filenames, img_h, img_w, classifier, isTrain, isOneHotEncod, dataAutoBalance, rateMagnifyData)
        if isTrain:
            print("total_train-datasets: ", labels.shape[0])
        else:
            print("total_test-datasets: ", labels.shape[0])

        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]


# classifier for labels
def classifier_labels(filenames, classifier):
    labels = []
    for filename in filenames:
        label = filename.split('/')[3]  # which folder is classifier
        labels.append(label)
    labels_k = list(set(labels))
    # print(labels)

    labels_count_dict = {}
    labels_dict = {}
    for label_k in labels_k:
        label_count_dict = {label_k: labels.count(label_k)}
        labels_count_dict.update(label_count_dict)

        # in binary case, only need to check label is same with classifier, if yes=1, no=0
        if label_k == classifier:
            label_v = 1
        else:
            label_v = 0
        label_dict = {label_k: label_v}
        labels_dict.update(label_dict)
    print(labels_count_dict)
    print(labels_dict)

    return labels, labels_count_dict, labels_dict


def get_one_hot_encoding(labels_v, NUM_CLASS):
    '''
    labels_v: class_value of labels, like: 1, 3, 2, 4... 
    CLASS_NUM: number of class
    '''
    labels_v = torch.Tensor(labels_v).unsqueeze(1).type(torch.int64)
    labels = torch.zeros(len(labels_v), NUM_CLASS).scatter_(1, labels_v, 1)
    # print(labels)

    return labels


def get_reshape_imgs(imgs):
    reshape_imgs = torch.tensor([], dtype=torch.uint8)
    for img in imgs:
        img = torch.from_numpy(img)
        img = torch.reshape(
            img, (img.shape[2], img.shape[0], img.shape[1]))
        img = img.unsqueeze(0)
        try:
            reshape_imgs = torch.cat((reshape_imgs, img), 0)
        except ValueError:
            reshape_imgs = img

    return reshape_imgs


# organising datas & labels
def organize_dataset(filenames, img_h, img_w, classifier, isTrain=False, isOneHotEncod=False, dataAutoBalance=True, rateMagnifyData=1.0):
    imgs = torch.tensor([], dtype=torch.uint8)
    # # labels = torch.tensor([], dtype=torch.int8)
    labels_v = torch.tensor([], dtype=torch.uint8)
    raw_labels, labels_count_dict, labels_dict = classifier_labels(
        filenames, classifier)

    # calulate populations of classifier_datas and unclassifier_datas
    num_classifier_datas = 0
    num_unclassifier_datas = 0
    for k, v in labels_count_dict.items():
        if k == classifier:
            num_classifier_datas = v
        else:
            num_unclassifier_datas += v
    incrace_ratio = num_unclassifier_datas / num_classifier_datas

    for i in range(len(filenames)):
        # data part: reshape -> reshape -> turn to tensor
        raw_img = cv2.imread(filenames[i])
        raw_img = cv2.resize(raw_img, (img_h, img_w))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # pre_imgs = torch.tensor([], dtype=torch.uint8)
        if dataAutoBalance is True:
            # let dataset be balance
            # use raw-img increase img by dip <<<Important_Part>>>
            if classifier == raw_labels[i] and incrace_ratio > 1:
                num_increase = round(incrace_ratio * rateMagnifyData) - 1
            elif classifier != raw_labels[i] and incrace_ratio < 1:
                num_increase = round((1 / incrace_ratio) * rateMagnifyData) - 1
            else:
                num_increase = round(rateMagnifyData) - 1
        else:
            num_increase = round(rateMagnifyData) - 1
        pre_imgs = dip_pre_process(raw_img, num_create=num_increase)

        for img in pre_imgs:
            img = torch.from_numpy(img)
            img = torch.reshape(
                img, (img.shape[2], img.shape[0], img.shape[1]))
            img = img.unsqueeze(0)
            try:
                imgs = torch.cat((imgs, img), 0)
            except ValueError:
                imgs = img
        # 合併的資料量一大速度就低落，待解決！！

        # label part: get classifier name
        label = raw_labels[i]

        for j in range(pre_imgs.shape[0]):
            label_v = torch.tensor([labels_dict.get(label)], dtype=torch.uint8)
            labels_v = torch.cat((labels_v, label_v), 0)
            if labels_v.shape[0] % 1000 == 0:
                print(
                    "Nunmber of already pre-process datasets: {}".format(len(labels_v)))

    if isOneHotEncod is True:
        NUM_CLASS = len(labels_dict)  # get number of class to one-hot-encoding
        labels = get_one_hot_encoding(labels_v, NUM_CLASS)
    else:
        labels = labels_v.to(torch.int64)

    return (imgs, labels)


if __name__ == "__main__":
    path = "Data/test"

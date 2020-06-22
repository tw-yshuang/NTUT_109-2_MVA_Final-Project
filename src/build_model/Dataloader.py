import cv2
import random
import torch
from find_file_name import *
from torch.utils.data import Dataset, DataLoader
from pre_process import dip_pre_process


class ImgDataset(Dataset):

    def __init__(self, path, isTrain=False):
        file_extension = "jpg"
        filenames = get_filenames(path, file_extension)
        random.shuffle(filenames)  # random sort files

        img_h = 128  # img resize hight
        img_w = 82  # img resize width

        (datas, labels) = organize_dataset(
            filenames, img_h, img_w, isTrain=isTrain, isOneHotEncod=False)
        print("total_train-datasets: ", labels.shape[0])

        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]


# classifer for labels
def classifer_labels(filenames):
    labels = []
    for filename in filenames:
        label = filename.split('/')[2]
        labels.append(label)
    labels = list(set(labels))
    # print(labels)

    labels_dict = {}
    for i in range(len(labels)):
        label_dict = {labels[i]: i}
        labels_dict.update(label_dict)
    print(labels_dict)

    return labels_dict


def get_one_hot_encoding(labels_v, NUM_CLASS):
    '''
    labels_v: class_value of labels, like: 1, 3, 2, 4... 
    CLASS_NUM: number of class
    '''
    labels_v = torch.Tensor(labels_v).unsqueeze(1).type(torch.int64)
    labels = torch.zeros(len(labels_v), NUM_CLASS).scatter_(1, labels_v, 1)
    # print(labels)

    return labels


# organising datas & labels
def organize_dataset(filenames, img_h, img_w, isTrain=False, isOneHotEncod=False):
    imgs = torch.tensor([], dtype=torch.uint8)
    # # labels = torch.tensor([], dtype=torch.int8)
    labels_v = []
    labels_dict = classifer_labels(filenames)
    for filename in filenames:
        # data part: reshape -> reshape -> turn to tensor
        raw_img = cv2.imread(filename)
        raw_img = cv2.resize(raw_img, (img_h, img_w))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        if isTrain is True:
            # use raw-img increase img by dip
            pre_imgs = dip_pre_process(raw_img, num_create=50)
            # # print(type(img))
            for img in pre_imgs:
                img = torch.from_numpy(img)
                img = torch.reshape(
                    img, (img.shape[2], img.shape[0], img.shape[1]))
                img = img.unsqueeze(0)
                try:
                    imgs = torch.cat((imgs, img), 0)
                except ValueError:
                    imgs = img
        else:
            img = raw_img

            img = torch.from_numpy(img)
            img = torch.reshape(
                img, (img.shape[2], img.shape[0], img.shape[1]))
            img = img.unsqueeze(0)
            try:
                imgs = torch.cat((imgs, img), 0)
            except ValueError:
                imgs = img

        # label part: get classifer name
        label = filename.split('/')[2]
        if isTrain is True:
            for i in range(pre_imgs.shape[0]):
                label_v = labels_dict.get(label)
                labels_v.append(label_v)
        else:
            label_v = labels_dict.get(label)
            labels_v.append(label_v)

    if isOneHotEncod is True:
        NUM_CLASS = len(labels_dict)  # get number of class to one-hot-encoding
        labels = get_one_hot_encoding(labels_v, NUM_CLASS)
    else:
        labels = torch.Tensor(labels_v).type(torch.int64)

    return (imgs, labels)


if __name__ == "__main__":
    path = "Data/test"
    dataloder = ImgDataset(path)
    # print(dataloder.labels)

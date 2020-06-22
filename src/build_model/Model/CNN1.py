import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from Dataloader import ImgDataset
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.nn = nn

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(256, 64, (3, 3), 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2)))

        self.ln1 = nn.Linear(88704, 256)
        self.out = nn.Linear(256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.ln1(x)
        x = self.out(x)
        # output = F.
        output = x

        return output

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = x.view(x.size(0), -1)
    #     self.out = nn.Linear(x.size(1), 10).to(device_0)
    #     output = self.out(x)

    #     return output

    # if isPoltSave is True:

    # test_output = cnn(test_datas[:10])
    # pred_y = torch.max(test_output, 1)[1].data.squeeze()
    # print(pred_y, 'prediction number')
    # print(test_labels[:10], 'real number')
if __name__ == "__main__":
    device_0 = get_device()

    train_path = "Data/train"
    train_datasets = ImgDataset(train_path, isTrain=True)

    test_path = "Data/test"
    test_datasets = ImgDataset(test_path)

    train_loader = Data.DataLoader(
        dataset=train_datasets,
        batch_size=6,
        shuffle=True,
        num_workers=2
    )

    cnn = CNN().to(device_0)
    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    EPOCH = 30
    train_model(EPOCH, train_loader, test_datasets)
    torch.save(cnn.state_dict(), 'out/cnn-model.pkl')

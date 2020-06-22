import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt


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

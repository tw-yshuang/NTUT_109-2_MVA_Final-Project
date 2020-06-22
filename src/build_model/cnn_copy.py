import torch
import torch.nn as nn
import torch.nn.functional as F
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


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    return device


def train_model(EPOCH, train_loader, test_datasets, isPoltSave=True):
    # test_datas = torch.tensor(test_datasets.datas, dtype=torch.float32)
    # test_labels = torch.tensor(test_datasets.labels, dtype=torch.int64)
    # test_datas = Variable(test_datas).to(device=device_0)
    # test_labels = Variable(test_labels).to(device=device_0)

    train_loss_ls = []
    train_acc_ls = []
    test_acc_ls = []
    for epoch in range(EPOCH):
        print("EPOCH: ", epoch)

        total = 0
        train_acc = 0
        for step, (datas, labels) in enumerate(train_loader):
            datas = torch.tensor(datas, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            b_datas = Variable(datas).to(device=device_0)
            b_labels = Variable(labels).to(device=device_0)

            output = cnn.forward(b_datas)
            loss = loss_func(output, b_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_train_labels = torch.max(output, 1)[1].data.squeeze()

            total += b_labels.size(0)

            # if step % 50 == 0:
            if step == len(labels):
                test_datas = torch.tensor(
                    test_datasets.datas, dtype=torch.float32)
                test_labels = torch.tensor(
                    test_datasets.labels, dtype=torch.int64)
                test_datas = Variable(test_datas).to(device=device_0)
                test_labels = Variable(test_labels).to(device=device_0)
                test_output = cnn(test_datas)
                pred_test_labels = torch.max(test_output, 1)[1].data.squeeze()

                # train_acc
                train_num_right = int(sum(pred_train_labels == b_labels))
                train_acc = train_num_right / b_labels.size(0)

                test_num_right = int(sum(pred_test_labels == test_labels))
                test_acc = test_num_right / test_labels.size(0)
                # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                print('Batch Size: {} | train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
                    step, train_acc, loss, test_acc))

                if loss > 1:
                    loss = 1
                train_loss_ls.append(loss)
                train_acc_ls.append(train_acc)
                test_acc_ls.append(test_acc)

    if isPoltSave is True:
        EPOCH_times = range(1, EPOCH+1)
        plt.plot(EPOCH_times, train_loss_ls, marker='o',
                 markerfacecolor='white', markersize=5)
        # 设置数字标签
        i = 0
        for a, b in zip(EPOCH_times, train_loss_ls):
            i += 1
            if i % 10 == 0 or i == EPOCH+1:
                if b != 1:
                    b = np.round((b.cpu()).detach().numpy(), 3)
                plt.text(a, b, b, ha='center',
                         va='bottom', fontsize=10)
        # 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
        plt.title("Train_loss", x=0.5, y=1.03)
        # 设置刻度字体大小
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # 標示x軸(labelpad代表與圖片的距離)
        plt.xlabel("Epoch", fontsize=10)
        # 標示y軸(labelpad代表與圖片的距離)
        plt.ylabel("Loss", fontsize=10)
        plt.savefig("out/train_loss.png")

        plt.cla()
        plt.plot(EPOCH_times, train_acc_ls, marker='o',
                 markerfacecolor='white', markersize=5)
        i = 0
        for d, e in zip(EPOCH_times, train_acc_ls):
            i += 1
            if i % 10 == 0 or i == EPOCH+1:
                if e != 1:
                    e = np.round((e.to('cpu')).detach().numpy(), 3)
                plt.text(d, e, e,
                         ha='center', va='bottom', fontsize=10)
        plt.plot(EPOCH_times, test_acc_ls, marker='o',
                 markerfacecolor='white', markersize=5)
        i = 0
        for a, b in zip(EPOCH_times, test_acc_ls):
            i += 1
            if i % 10 == 0 or i == EPOCH+1:
                if b != 1:
                    b = np.round(b, 3)
                plt.text(a, b, b,
                         ha='center', va='top', fontsize=10)
        # 設定圖片標題，以及指(定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
        plt.title("Accuracy", x=0.5, y=1.03)
        # 设置刻度字体大小
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # 標示x軸(labelpad代表與圖片的距離)
        plt.xlabel("Epoch", fontsize=10)
        # 標示y軸(labelpad代表與圖片的距離)
        plt.ylabel("Accuracy", fontsize=10)
        plt.savefig("out/acc.png")

    # test_output = cnn(test_datas[:10])
    # pred_y = torch.max(test_output, 1)[1].data.squeeze()
    # print(pred_y, 'prediction number')
    # print(test_labels[:10], 'real number')


if __name__ == "__main__":
    device_0 = get_device()
    torch.cuda.empty_cache()

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

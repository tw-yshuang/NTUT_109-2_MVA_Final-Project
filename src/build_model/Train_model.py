import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Model.CNN import CNN

# model = CNN().to('cuda:0')


def train_model(device, EPOCH, train_loader, test_datasets, model=CNN().to('cuda:0'), printHistory=True):

    # model = CNN().to('cuda:0')
    # test_datasets import to the device
    test_datas = test_datasets.datas.clone().detach().type(torch.float32)
    test_labels = test_datasets.labels.clone().detach().type(torch.int64)
    test_datas = Variable(test_datas).to(device)
    test_labels = Variable(test_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    train_loss_ls = []
    train_acc_ls = []
    test_acc_ls = []
    for epoch in range(1, EPOCH+1):
        print("EPOCH: ", epoch)

        total = 0
        train_acc = 0
        for step, (datas, labels) in enumerate(train_loader):
            datas = datas.clone().detach().type(torch.float32)
            labels = labels.clone().detach().type(torch.int64)
            b_datas = Variable(datas).to(device)
            b_labels = Variable(labels).to(device)

            output = model(b_datas)
            loss = loss_func(output, b_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_train_labels = torch.max(output, 1)[1].data.squeeze()
            # pred_train_labels = F.log_softmax(output, dim=1)

            total += b_labels.size(0)

            # if step % 50 == 0:
            if step == len(labels):
                # test_datas = torch.tensor(
                #     test_datasets.datas, dtype=torch.float32)
                # test_labels = torch.tensor(
                #     test_datasets.labels, dtype=torch.int64)
                # test_datas = Variable(test_datas).to(device)
                # test_labels = Variable(test_labels).to(device)
                test_output = model(test_datas)

                pred_test_labels = torch.max(test_output, 1)[1].data.squeeze()
                # pred_test_labels = F.log_softmax(test_output, dim=1)

                # train_acc
                train_num_right = int(sum(pred_train_labels == b_labels))
                train_acc = train_num_right / b_labels.size(0)

                test_num_right = int(sum(pred_test_labels == test_labels))
                test_acc = test_num_right / test_labels.size(0)
                # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                if printHistory is True:
                    print('Batch Size: {} | train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
                        step, train_acc, loss, test_acc))

                if loss > 1:
                    loss = 1
                train_loss_ls.append(loss)
                train_acc_ls.append(train_acc)
                test_acc_ls.append(test_acc)

    return (train_loss_ls, train_acc_ls, test_acc_ls)

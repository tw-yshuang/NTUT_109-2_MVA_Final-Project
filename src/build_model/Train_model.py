import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Model.CNN import CNN


def test_performance(model, loss, test_datas, pred_train_labels, b_labels, test_labels, printHistory=True):
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
        print('train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
            train_acc, loss, test_acc))

    return train_acc, test_acc


def train_model(device, EPOCH, train_loader, test_datasets, model=CNN().to('cuda:0'), printHistory=True):

    # model = CNN().to('cuda:0')
    # test_datasets import to the device
    test_datas = test_datasets.datas.clone().detach().type(torch.float32)
    test_labels = test_datasets.labels.clone().detach().type(torch.int64)
    test_datas = Variable(test_datas, requires_grad=False).to(device)
    test_labels = Variable(test_labels, requires_grad=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    train_loss_ls = []
    train_acc_ls = []
    test_acc_ls = []
    for epoch in range(1, EPOCH+1):
        # print("EPOCH: ", epoch)

        total = 0
        train_acc = 0
        for step, (datas, labels) in enumerate(train_loader):
            datas = datas.clone().detach().type(torch.float32)
            labels = labels.clone().detach().type(torch.int64)
            b_datas = Variable(datas).to(device)
            b_labels = Variable(labels).to(device)

            output = model.forward(b_datas)
            loss = loss_func(output, b_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_train_labels = torch.max(output, 1)[1].data.squeeze()
            # pred_train_labels = F.log_softmax(output, dim=1)

            total += b_labels.size(0)

            # if step % 10 == 0:
            #     train_acc, test_acc = test_performance(model, loss, test_datas,
            #                                            pred_train_labels, b_labels, test_labels, printHistory=False)

            #     if printHistory is True:
            #         print('Step: {} | train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
            #             step, train_acc, loss, test_acc))

        train_acc, test_acc = test_performance(model, loss, test_datas,
                                               pred_train_labels, b_labels, test_labels, printHistory=False)

        if printHistory is True:
            print('EPOCH: {} | train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
                epoch, train_acc, loss, test_acc))

        if loss > 1:
            loss = 1

        try:
            train_loss_ls.append(float(loss.cpu().detach().numpy()))
        except AttributeError:
            train_loss_ls.append(float(loss))
        train_acc_ls.append(train_acc)
        test_acc_ls.append(test_acc)

    return (train_loss_ls, train_acc_ls, test_acc_ls)

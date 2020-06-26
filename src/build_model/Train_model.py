import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Model.CNN import CNN


def test_performance(model, device, test_loader, loss, pred_train_labels, b_labels, printHistory=True):
    test_num_right = 0
    for step, (datas, labels) in enumerate(test_loader):
        b_test_datas = datas.clone().detach().type(torch.float32)
        b_test_labels = labels.clone().detach().type(torch.int64)
        b_test_datas = Variable(b_test_datas).to(device)
        # b_test_labels don't need to calculate loss
        b_test_labels = Variable(b_test_labels)

        b_test_output = model(b_test_datas)
        pred_test_labels = torch.max(b_test_output, 1)[1].data.squeeze().cpu()

        # pred_test_labels = F.log_softmax(test_output, dim=1)
        b_test_num_right = int(sum(pred_test_labels == b_test_labels))
        test_num_right += b_test_num_right

    # train_acc
    train_num_right = int(sum(pred_train_labels == b_labels))
    train_acc = train_num_right / b_labels.size(0)

    # test_acc
    test_acc = test_num_right / len(test_loader.dataset.labels)

    if printHistory is True:
        print('train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
            train_acc, loss, test_acc))

    return train_acc, test_acc


def train_model(device, EPOCH, train_loader, test_loader, model, loss_func, optimizer, printHistory=True):

    train_loss_ls = []
    train_acc_ls = []
    test_acc_ls = []
    for epoch in range(1, EPOCH+1):
        # print("EPOCH: ", epoch)

        total = 0
        train_acc = 0
        for step, (datas, labels) in enumerate(train_loader):
            b_datas = datas.clone().detach().type(torch.float32)
            b_labels = labels.clone().detach().type(torch.int64)
            b_datas = Variable(b_datas).to(device)
            b_labels = Variable(b_labels).to(device)

            output = model.forward(b_datas)
            loss = loss_func(output, b_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_train_labels = torch.max(output, 1)[1].data.squeeze().cpu()
            # pred_train_labels = F.log_softmax(output, dim=1)

            total += b_labels.size(0)

            # if step % 10 == 0:
            #     train_acc, test_acc = test_performance(model, loss, test_datas,
            #                                            pred_train_labels, b_labels, test_labels, printHistory=False)

            #     if printHistory is True:
            #         print('Step: {} | train_acc: {:5f} | train_loss: {:5f} | test_acc: {:5f}'.format(
            #             step, train_acc, loss, test_acc))

        train_acc, test_acc = test_performance(model, device, test_loader, loss.cpu(),
                                               pred_train_labels, b_labels.cpu(), printHistory=False)

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

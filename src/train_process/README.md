# Process of train, 訓練流程

---

## 1. Train-data_classifer

1. Split train-data  
   from _`doc/train.csv`_'s columns `ImageID` & `ClassID` ,split 5 class(4 defect ＆ 1 ok) imgs in to 5 different folders, let datasets get more easlier to unstand.

2. Select-encodepart  
   from _`doc/train.csv`_'s columns `ImageID` & `ClassID` ＆ `EncodedPixels`, use `EncodedPixels`'s information to turn encod_pixel into the img, and find correpond img use dip way to get bbox_imgs and save it, be our train_datasets.

## 2. Build & Train CNN_model

1.  Buid `Dataloader.py`  
    build a function call : "`organize_dataset`",

    > Function : `organize_dataset`
    >
    > > input:
    > >
    > > > filenames,  
    > > > img_h, img_w,  
    > > > classifier,  
    > > > isTrain=False,  
    > > > isOneHotEncod=False,  
    > > > dataAutoBalance=True,  
    > > > rateMagnifyData=1.0
    > >
    > > output:
    > >
    > > > datas,  
    > > > labels

    it can auto scan and import imgs from the path you input, and organize datas (e.g. rateMagnifyData, autoBalance, pre_process...), labels (label is from the folder name under the path)

2.  Buid `Train_model.py`  
    import datasets that already setup from `Dataloader.py`, and split by **batch_size**, step by step to train model, and save all history(**EPOCE , train_loss, train_acc, test_acc**) information.

3.  Buid `CNN.py`

    > layer 1 :nn.Conv2d
    >
    > > in_channels=3, out_channels=32,
    > > kernel_size=(5, 5),
    > > stride=1, padding=2,
    > > nn.ReLU(),
    > > nn.MaxPool2d(kernel_size=(2, 2))

    > layer 2 :nn.Conv2d
    >
    > > in_channels=32, out_channels=128,
    > > kernel_size=(3, 3),
    > > stride=1, padding=2,
    > > nn.ReLU(),
    > > nn.MaxPool2d(kernel_size=(2, 2))

    > layer 3 :nn.Linear
    >
    > > in_channels=36992,  
    > > out_channels=128

    > layer 4 :nn.Linear
    >
    > > in_channels=128,  
    > > out_channels=32

    > layer 5 :nn.Linear
    >
    > > in_channels=32,  
    > > out_channels=2

4.  Buid `main.py`  
    all process are merge in here, including most of customer settings.

5.  NOW, the structure of train cnn model are get ready, train!!

## 3. Build SVM_model(for HOG)

## 4. done !!

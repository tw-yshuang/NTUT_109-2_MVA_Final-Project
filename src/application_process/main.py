import cv2
import torch
import random
import pandas as pd
from torch.autograd import Variable
from Model.HOG_find_encode import hog_find_encode
from Model.class1_DIP import get_c1_img
from Model.class2_DIP import get_c2_img
from Model.class3_DIP import get_c3_img
from Model.class4_DIP import get_c4_img
from ANS import ans_img
# from Model.CNN_3 import CNN


class num_class_defect(object):
    def __init__(self):
        super().__init__()
        self.imgs = []
        self.positions = []


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    return device


def model_upload(device):
    class_1_CNN = 'out/class_1/cnn3_b2048_e300/cnn-model_compelet.pkl'
    class_2_CNN = 'out/class_2/cnn3_b2048_e300/cnn-model_compelet.pkl'
    class_3_CNN = 'out/class_3/m2_cnn3_b2048_e92/m2_cnn3_b2048_e92/cnn-model_compelet.pkl'
    class_4_CNN = 'out/class_4/m3_cnn3_b2048_e92/cnn-model_compelet.pkl'

    torch.cuda.empty_cache()
    # class 1 model
    model_1 = torch.load(class_1_CNN).to(device)
    model_1.eval()

    # class 2 model
    model_2 = torch.load(class_2_CNN).to(device)
    model_2.eval()

    # class 3 model
    model_3 = torch.load(class_3_CNN).to(device)
    model_3.eval()

    # class 4 model
    model_4 = torch.load(class_4_CNN).to(device)
    model_4.eval()

    return [model_1, model_2, model_3, model_4]


def reshape_tensor_from_np(raw_img, img_h, img_w):
    img = cv2.resize(raw_img, (img_h, img_w))
    img = torch.from_numpy(img)
    img = torch.reshape(
        img, (img.shape[2], img.shape[0], img.shape[1]))
    img = img.unsqueeze(0)
    return img


if __name__ == "__main__":
    df = pd.read_csv('doc/train.csv')
    imgs_path = 'Data/train_images'
    save_file_path = 'select-encode_part'

    device = get_device()
    models = model_upload(device)

    for num_img in range(0, 1000):
        num_img = random.randint(0, 1000) + df.shape[0] - 1000
        img_name = df.iloc[num_img, 0]
        folder_name = df.iloc[num_img, 1]
        img_path = '{}/{}/{}'.format(imgs_path, folder_name, img_name)
        total_crop_imgs, total_crop_postions = hog_find_encode(img_path)
        img = cv2.imread(img_path)

        c1 = num_class_defect()
        c2 = num_class_defect()
        c3 = num_class_defect()
        c4 = num_class_defect()
        class_defect = [c1, c2, c3, c4]

        # first shape is use to separate multiple class from one raw_img
        for i in range(len(total_crop_imgs)):
            crop_imgs = total_crop_imgs[i]
            crop_positions = total_crop_postions[i]

            # second shape is for crop_imgs
            for j in range(len(crop_imgs)):
                crop_img = reshape_tensor_from_np(
                    crop_imgs[j], img_h=64, img_w=64).detach().type(torch.float32)
                crop_img = Variable(crop_img, requires_grad=False).to(device)

                # use every model to predict
                for k in range(len(models)):
                    output = models[k](crop_img)
                    predict = torch.argmax(output, dim=1).detach().cpu()
                    if predict == 1:
                        class_defect[k].imgs.append(crop_imgs[j])
                        class_defect[k].positions.append(crop_positions[j])

        c1_img = get_c1_img(img.copy(), c1.positions)
        c2_img = get_c2_img(img.copy(), c2.positions)
        c3_img = get_c3_img(img.copy(), c3.positions)
        c4_img = get_c4_img(img.copy(), c4.positions)

        c_imgs = [c1_img, c2_img, c3_img, c4_img]
        for i in range(1, 5):
            cv2.namedWindow('{}'.format(i), 0)
            cv2.resizeWindow('{}'.format(
                i), (img.shape[1] // 1, img.shape[0] // 1))
            cv2.moveWindow('{}'.format(i), 2000, 400*i)
            cv2.imshow('{}'.format(i), c_imgs[i-1])

        for i in range(1, 5):
            cv2.destroyWindow('A_{}'.format(i))
            # cv2.namedWindow('A_{}'.format(i), i+3)
            # cv2.resizeWindow('A_{}'.format(
            #     i), (img.shape[1]//2, img.shape[0]//2))
        ans_img(img_path, isShow=True)

        cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

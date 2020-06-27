# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.is_available())

# import os
# folder_path = 'Data/train_images/select-encode_part/5'

# if os.path.exists(folder_path) is False:
#     os.makedirs(folder_path)
# else:
#     print('132')


# def create_img():
#     img_w = img[i].shape[0]
#     img_h = img[i].shape[0]
#     if abs(img_w - img_h) > 5:
#         if img_w > img_h:
#             square_img = np.zeros(
#                 (img_w, img_w), dtype='uint8')
#             x_center = img_w // 2
#             h_half = img_h / 2
#             if h_half is int:
#                 square_img[0:img_w, img_w -
#                            h_half: img_w + h_half] = img[i]
#             else:
#                 h_half = round(h_half)
#                 square_img[0:img_w, img_w -
#                            h_half: img_w + h_half+1] = img[i]

import cv2
path = 'Data/train_images/select-encode_part/3/0a405b396_2.jpg'
img = cv2.imread(path)
print(img.shape)
cv2.imshow('img', img)

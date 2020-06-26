# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.is_available())

import os
folder_path = 'Data/train_images/select-encode_part/5'

if os.path.exists(folder_path) is False:
    os.makedirs(folder_path)
else:
    print('132')

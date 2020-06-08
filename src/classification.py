import os
import shutil
import pandas as pd
'''
This program is classification datas for this project,
let who want to analysis datas can more easier to unstand datas.
'''


def import_table(path, import_cols=None):
    df = pd.read_csv(path, encoding="utf-8")  # mbcs, utf-8

    if import_cols is not None:
        df = df.loc[:, import_cols]

    return df


if __name__ == "__main__":
    path = 'doc/train.csv'
    cols = ['ImageId', 'ClassId']
    df = import_table(path, cols)
    for i in range(df.shape[0]):
        img_path = df.loc[i, cols[0]]  # image_path
        try:
            shutil.copyfile("data/train_images/raw/{}".format(img_path),
                            "data/train_images/{}/{}".format(df.loc[i, cols[1]], img_path))
        except OSError:
            try:
                dir_path = os.path.abspath('.')
                dir_path += "/data/train_images/{}".format(
                    df.loc[i, cols[1]])
                os.mkdir(dir_path)
                print("create a file, path: {}".format(dir_path))

                shutil.copyfile("data/train_images/raw/{}".format(img_path),
                                "data/train_images/{}/{}".format(df.loc[i, cols[1]], img_path))
            except OSError:
                continue
    print("done!")

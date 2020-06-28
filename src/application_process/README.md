# Process of application, 應用流程

---

## 1. read raw_img

>

## 2. hog dectection raw_imgs

> `API`
>
> > input: raw_img  
> > output: hog_imgs

> ideal is use hog to find the encod part, but now, our **SVM** model is even not already to train yet, so there come up **_Plan_A_** and **_Plan_B_** :

- **_Plan_A_** : for ideal

  > use hog to find the encod part

- **_Plan_B_** : for now
  > because this final project is decided to use less 1000 imgs from _`train.csv`_ to become our application process test, and all of test_part have encod_pixel information, so for now, we use select-encode_part (from file: train-data_classifier) to pretend our hog dectection temporary.

## 3. resize every hog_img

1. one by one let model predict hog_img and get conclusion

2. each conclusion correspond each dip way to get the encoding_part

## 4. merge every encoding_part back to the raw_img

>

## 5. turn the raw_img with encoding_part to the encod_pixel csv

>

## 6. done !!

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from operator import itemgetter
import operator
import os


def segm_label(input_img, rgb_img):
    (h, w) = input_img.shape[:2]
    print(h,w)

    "Return a list containing the sum of the pixels in each column"
    sumCols = []
    for j in range(w):
        col = 255 - input_img[0:h, j:j + 1]
        sumCols.append(np.sum(col/255))

    "Return a list containing the sum of the pixels in each row"
    sumRows = []
    for j in range(h):
        row = 255 - input_img[j:j + 1, 0:w]
        sumRows.append(np.sum(row / 255))
        
    cur_obj = 1
    yy = [1]

    for i in range(len(sumCols)):
        if sumCols[i] > 120:
            if i - cur_obj > 16:
                yy.append(i)
                cur_obj = i
            else:
                cur_obj = i

    # draw a red bordered rectangle over the character.
    cur = yy[0]
    for x in yy:
        if x - cur < 5:
            cur = x
        else:
            x =x
        cv2.rectangle(rgb_img, (cur, 0), (x + cur, h), (255, 0, 0), 1)
        
    # cv2.imshow('ima', input_img)
    # cv2.waitKey(0)

    # fig, arr = plt.subplots(2, 2)
   

    # arr[0][0].imshow(input_img, cmap="gray")
    # arr[0][1].imshow(rgb_img, cmap="gray")
    # arr[1][1].plot(sumCols)
    # arr[1][0].plot(sumRows)
    # # arr[0][1].imshow(plate_segm, cmap="gray")

    # plt.show()

    return input_img, sumCols


dir = os.path.dirname(__file__)
# input_img_uri = os.path.join(dir, '../input_imgs/plate.png')
image_uri = os.path.join(dir, '../images/plate_cropped.png')
img = cv2.imread(image_uri)

segm_label(img, img)
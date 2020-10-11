import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

def cropp_by_projection(input_img):
    input_img = cv2.resize(input_img, (382, 84))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equal_histogram = clahe.apply(input_img)

    ret, invert_input_img = cv2.threshold(equal_histogram, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h = invert_input_img.shape[0]
    w = invert_input_img.shape[1]

    kernel = np.ones((3, 5), np.uint8)
    img_close = cv2.morphologyEx(invert_input_img, cv2.MORPH_CLOSE, kernel)

    "Return a list containing the sum of the pixels in each column"
    (h, w) = img_close.shape[:2]
    sumCols = []
    for j in range(w):
        col = 255 - img_close[0:h, j:j + 1]
        sumCols.append(np.sum(col / 255))

    "Return a list containing the sum of the pixels in each row"
    sumRows = []
    for j in range(h):
        row = 255 - img_close[j:j + 1, 0:w]
        sumRows.append(np.sum(row / 255))

    left_line = 0
    right_line = w
    top_line = 0
    bottom_line = h

    left_l = 0
    left_none = []
    while left_l < w * 0.25:
        if sumCols[left_l] >= h * 0.7:
            left_none = []
            left_line = left_l
        else:
            left_line = left_l
            left_none.append(left_line)
            if len(left_none) > 4:
                left_line = left_none[0]
                break
        left_l += 1

    right_l = w - 1
    right_none = []
    while right_l > w * 0.75:
        if sumCols[right_l] >= h * 0.7:
            right_none = []
            right_line = right_l
        else:
            right_line = right_l
            right_none.append(right_line)
            if len(right_none) > 4:
                right_line = right_none[0]
                break
        right_l -= 1

    top_l = h//2
    while top_l > 0:
        if sumRows[top_l] <= w * 0.75:
            top_line = top_l
        else:
            top_line = top_l
            break
        top_l -= 1

    bottom_l = h//2
    while bottom_l < h:
        if sumRows[bottom_l] <= w * 0.75:
            bottom_line = bottom_l
        else:
            bottom_line = bottom_l
            break
        bottom_l += 1

    if top_line == bottom_line or left_line == right_line:
        return None

    cropped_plate = input_img[top_line:bottom_line, left_line:right_line]
    
    # fig, axarr = plt.subplots(2, 2)

    # axarr[0, 0].imshow(invert_input_img, cmap="gray");
    # axarr[1, 1].imshow(cropped_plate, cmap="gray");
    # axarr[0, 1].plot(sumCols);
    # axarr[1, 0].plot(sumRows);
    # plt.show()

    invert_cropped_plate2_resized = cv2.resize(cropped_plate, (382, 84))

    invert_cropped_plate = cv2.copyMakeBorder(invert_cropped_plate2_resized, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    return invert_cropped_plate

import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

def setup_axarr(fig, rect, rotation, axisScale, axisLimits, doShift):
    tr_rot = Affine2D().scale(axisScale[0], axisScale[1]).rotate_deg(rotation)

    # This seems to do nothing
    if doShift:
        tr_trn = Affine2D().translate(-90,-5)
    else:
        tr_trn = Affine2D().translate(0,0)

    tr = tr_rot + tr_trn

    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=axisLimits)

    ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax)
    aux_ax = ax.get_aux_axes(tr)

    return ax, aux_ax


def cropp_by_projection(input_image):
    # ! optional
    # input_image = cv2.resize(input_image, (382, 84))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equal_histogram = clahe.apply(input_image)

    ret, invert_input_image = cv2.threshold(equal_histogram, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h = invert_input_image.shape[0]
    w = invert_input_image.shape[1]

    kernel = np.ones((3, 5), np.uint8)
    dilate = cv2.morphologyEx(invert_input_image, cv2.MORPH_CLOSE, kernel)

    # def verticalProjection(img):
    "Return a list containing the sum of the pixels in each column"
    (h, w) = dilate.shape[:2]
    sumCols = []
    for j in range(w):
        col = 255 - dilate[0:h, j:j + 1]
        sumCols.append(np.sum(col / 255))

    "Return a list containing the sum of the pixels in each column"
    sumRows = []
    for j in range(h):
        row = 255 - dilate[j:j + 1, 0:w]  # y1:y2, x1:x2
        sumRows.append(np.sum(row / 255))

    hh = []
    for i in range(h):
        hh.append(i)

    ww = []
    for i in range(w):
        ww.append(i)

    left_line = 0
    right_line = w
    top_line = 0
    bottom_line = h

    left_l = 0
    left_none = []
    while left_l < w / 6:
        # if sumCols[left_l] >= h * 0.65:
        if sumCols[left_l] >= h * 0.75:
            left_none = []
            left_line = left_l
            # continue
        else:
            left_line = left_l
            left_none.append(left_line)
            if len(left_none) > 2:
                left_line = left_none[0]
                break
        left_l += 1

    right_l = w - 1
    right_none = []
    while right_l > w / 1.8:
        # if sumCols[right_l] >= h * 0.65:
        if sumCols[right_l] >= h * 0.75:
            right_none = []
            right_line = right_l
            # continue
        else:
            right_line = right_l
            right_none.append(right_line)
            if len(right_none) > 4:
                right_line = right_none[0]
                break
        right_l -= 1

    top_l = h//2
    # print(sumRows[top_l])
    while top_l > 0:
        if sumRows[top_l] <= w * 0.75:
            top_line = top_l
            # continue
        else:
            top_line = top_l
            break

        top_l -= 1

    bottom_l = h//2
    while bottom_l < h:
        if sumRows[bottom_l] <= w * 0.75:
            bottom_line = bottom_l
            # continue
        else:
            bottom_line = bottom_l
            break
        bottom_l += 1

    if top_line == bottom_line or left_line == right_line:
        return None

    # cropped_plate = invert_input_image[top_line: bottom_line, left_line:right_line]
    cropped_plate = input_image[top_line: bottom_line, left_line:right_line]
    # cropped_input_plate = input_image[top_line: bottom_line, left_line:right_line]
    ret, cropped_input_plate = cv2.threshold(cropped_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('aa', cropped_plate)
    # cv2.waitKey(0)
    # ret, invert_cropped_plate = cv2.threshold(cropped_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # invert_cropped_plate = cv2.adaptiveThreshold(cropped_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

    # print(sumRows[top_l])
    (h2, w2) = cropped_input_plate.shape[:2]
    left_line2 = 0
    right_line2 = w2
    top_line2 = 0
    bottom_line2 = h2
    hh2 = []
    for i in range(h2):
        hh2.append(i)

    ww2 = []
    for i in range(w2):
        ww2.append(i)

    sumCols2 = []
    for j in range(w2):
        col = 255 - cropped_input_plate[0:h2, j:j + 1]  # y1:y2, x1:x2
        # print(col)
        sumCols2.append(np.sum(col / 255))
    sumRows2 = []
    for j in range(h2):
        # row = img[0:h, j:j+1] # y1:y2, x1:x2
        row = 255 - cropped_input_plate[j:j + 1, 0:w2]  # y1:y2, x1:x2
        sumRows2.append(np.sum(row / 255))

    left_l2 = 0
    left_none2 = []
    while left_l2 < w2 / 6:
        # if sumCols[left_l] >= h * 0.65:
        if sumCols2[left_l2] >= h2 * 0.85:
            left_none2 = []
            left_line2 = left_l2
            # continue
        else:
            left_line2 = left_l2
            left_none2.append(left_line2)
            if len(left_none2) > 2:
                left_line2 = left_none2[0]
                break
        left_l2 += 1

    right_l2 = w2 - 1
    right_none2 = []
    while right_l2 > w2 / 1.8:
        # if sumCols[right_l] >= h * 0.65:
        if sumCols2[right_l2] >= h2 * 0.85:
            right_none2 = []
            right_line2 = right_l2
            # continue
        else:
            right_line2 = right_l2
            right_none2.append(right_line2)
            if len(right_none2) > 2:
                right_line2 = right_none2[0]
                break
        right_l2 -= 1

    top_l2 = 0
    top_none2 = []
    # print(sumRows[top_l])
    while top_l2 < h2//6:
        if sumRows2[top_l2] >= w2 * 0.6:
            top_none2 = []
            top_line2 = top_l2
            # continue
        else:
            top_line2 = top_l2
            top_none2.append(top_line2)
            if len(top_none2) > 2:
                top_line2 = top_none2[0]
                break

        top_l2 += 1

    # top_l2 = 0
    # while top_l2 < h2 / 4:
    #     if sumRows2[top_l2] >= w2 * 0.75:
    #         top_line2 = top_l2
    #         # continue
    #     top_l2 += 1
    # print('top_line2', top_line2)

    bottom_l2 = h2
    bottom_none2 = []
    # print(sumRows[top_l])
    while bottom_l2 < h2 // 1.2:
        if sumRows2[bottom_l2] >= w2 * 0.6:
            bottom_none2 = []
            bottom_line2 = bottom_l2
            # continue
        else:
            bottom_line2 = bottom_l2
            bottom_none2.append(bottom_line2)
            if len(bottom_none2) > 2:
                bottom_line2 = bottom_none2[0]
                break

        bottom_l2 -= 1

    invert_cropped_plate = cv2.bitwise_not(cropped_input_plate)

    
    kernel = np.ones((3, 3), np.uint8)
    erosed_image = cv2.erode(invert_cropped_plate, kernel)  

    invert_cropped_plate2 = erosed_image[top_line2: bottom_line2, left_line2:right_line2]

    invert_cropped_plate2_resized = cv2.resize(invert_cropped_plate2, (382, 84))

    invert_cropped_plate = cv2.copyMakeBorder(invert_cropped_plate2_resized, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    # fig = plt.figure(1, figsize=(8, 8))
    # axarr = [fig.add_subplot(221),fig.add_subplot(223), fig.add_subplot(222), fig.add_subplot(224)]

    # # fig, axarr = plt.subplots(2, 2)


    # ax, aux_ax = setup_axarr(fig, 222, 270, 
    #                         [1,-1], (0, h, 0, np.max(sumRows)), False)
    # axarr[2] = aux_ax
    # label_axes = ax

    # for axisLoc in ['top','left','right']:
    #     label_axes.axis[axisLoc].set_visible(False)
    # label_axes.axis['bottom'].toggle(ticklabels=False)

    # axarr[0].imshow(invert_input_image, cmap="gray");
    # axarr[1].plot(sumCols);
    # axarr[2].plot(sumRows);
    # axarr[3].imshow(dilate, cmap="gray");
    
    # fig.subplots_adjust(wspace=0.20, hspace=0.20, left=0.05, right=0.99, top=0.96, bottom=0.0)
    # plt.show()

    return invert_cropped_plate

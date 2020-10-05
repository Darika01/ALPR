import cv2
import numpy as np
from skimage.transform import resize


def cropp_by_haaf_lines(input_img):

    # img = (input_img).astype('uint8')
    # ret, invert_input_image = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    invert_input_image = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

    # invert_input_image = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

    h = invert_input_image.shape[0]
    w = invert_input_image.shape[1]

    # invert_input_image = cv2.bitwise_not(img)
    kernel = np.ones((1, 13), np.uint8)

    dilate = cv2.morphologyEx(invert_input_image, cv2.MORPH_CLOSE, kernel)

    canny_image = cv2.Canny(dilate, 0, 200)
    # cv2.imshow('binary_plate2',canny_image)

    left_lines = [(0, 0, w, h)]
    right_lines = [(w, h, 0, 0)]
    top_lines = [(0, 0, w, h)]
    bottom_lines = [(w, h, 0, 0)]

    minLineLength = 14
    maxLineGap = 15
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 1, minLineLength, maxLineGap)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                # print(angle)
                if angle == 0:
                    if y1 < h / 5 and y2 < h / 5:
                        if y1 == 0:
                            top_lines.append((x1, y1 + 1, x2, y2))
                        else:
                            top_lines.append((x1, y1, x2, y2))
                        # cv2.line(grayscale_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    elif y1 > h / 1.5 and y2 > h / 1.5:
                        # else:
                        bottom_lines.append((x1, y1, x2, y2))
                        # cv2.line(grayscale_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def sortFirst(val):
        return val[0]

    # function to return the second element of the
    # two elements passed as the paramater
    def sortSecond(val):
        return val[1]

    # sorts the array in ascending according to second element
    if len(top_lines) > 1:
        top_lines.sort(key=sortSecond, reverse=True)
        # print('top_lines', top_lines)

    # sorts the array in ascending according to second element
    if len(bottom_lines) > 1:
        bottom_lines.sort(key=sortSecond)
        # print('bottom_lines', bottom_lines)

    cropped_plate1 = invert_input_image[top_lines[0][1]:bottom_lines[0][1], 0: w]
    # plt.imshow(binary_plate2, cmap="gray")
    # plt.show()
    # cv2.imshow('binary_plate2',canny)

    h2 = cropped_plate1.shape[0]
    w2 = cropped_plate1.shape[1]

    kernel2 = np.ones((13, 3), np.uint8)

    dilate2 = cv2.morphologyEx(cropped_plate1, cv2.MORPH_CLOSE, kernel2)
    # dilate2 = cv2.morphologyEx(cropped_plate1, cv2.MORPH_OPEN, kernel2)
    canny_image2 = cv2.Canny(dilate2, 0, 200)
    # inverted = cv2.bitwise_not(dilate2)

    kernel3 = np.ones((1, 5), np.uint8)

    # canny_image2 = cv2.morphologyEx(canny_image2, cv2.MORPH_CLOSE, kernel3)

    minLineLength2 = h2 / 1.4
    maxLineGap2 = 4
    lines2 = cv2.HoughLinesP(canny_image2, 1, np.pi / 180, 3, minLineLength2, maxLineGap2)
    if lines2 is not None:
        for line in lines2:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                # print(angle)
                if angle == 90 or angle == -90:
                    if x1 < w / 7:
                        left_lines.append((x1, y1, x2, y2))
                    elif x1 > w / 1.2:
                        right_lines.append((x1, y1, x2, y2))
                    # cv2.line(grayscale_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if len(left_lines) > 1:
        left_lines.sort(key=sortFirst, reverse=True)
        # print('left_lines', left_lines)
        while (left_lines[0][0] - left_lines[1][0] < 4) or (left_lines[0][1] - left_lines[0][3] < h2 / 3.2):
            del left_lines[0]
            if len(left_lines) == 1:
                break
            # return right_lines
        # print('left_lines', left_lines)

    if len(right_lines) > 1:
        right_lines.sort(key=sortFirst)
        # right_lines.sort(key=sortFirst, reverse=True)
        # print('right_lines', right_lines)

        while (right_lines[1][0] - right_lines[0][0] > 5) or (right_lines[0][1] - right_lines[0][3] < h2 / 2):
            del right_lines[0]
            if len(right_lines) == 1:
                break
            # return right_lines
        # print('right_lines', right_lines)

        # if right_lines[1][0] - right_lines[0][0] < 7:
        #     del right_lines[0]
        # print('right_lines', right_lines)

    # if len()
    cropped_plate = cropped_plate1[0: h, left_lines[0][0]:right_lines[0][0]]

    cropped_plate = cv2.bitwise_not(cropped_plate)

    #
    # binary_plate2 = cv2.copyMakeBorder(binary_plate2, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)
    cropped_plate = cv2.copyMakeBorder(cropped_plate, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    # crop_img = cv2.copyMakeBorder(crop_img, 1,1,1,1, cv2.BORDER_CONSTANT, None, (255,255,255))
    # binary_plate2 = np.uint8(cropped_plate)

    return cropped_plate

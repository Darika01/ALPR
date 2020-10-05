import cv2
import numpy as np
from skimage.transform import resize


def cropp_by_haaf_lines(input_img):

    # img = (input_img).astype('uint8')
    # ret, invert_input_image = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert_input_image = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equal_histogram = clahe.apply(input_img)

    ret, invert_input_image = cv2.threshold(equal_histogram, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert_input_image = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

    h = invert_input_image.shape[0]
    w = invert_input_image.shape[1]

    # invert_input_image = cv2.bitwise_not(img)
    kernel = np.ones((1, 15), np.uint8)

    dilate = cv2.morphologyEx(invert_input_image, cv2.MORPH_CLOSE, kernel)

    canny_image = cv2.Canny(dilate, 0, 200)
    # cv2.imshow('binary_plate2',canny_image)

    left_lines = [(0, 0, w, h)]
    right_lines = [(w, h, 0, 0)]
    top_lines = [(0, 0, w, h)]
    bottom_lines = [(w, h, 0, 0)]

    minLineLength = h / 1.4
    maxLineGap = 2
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 1, minLineLength, maxLineGap)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if -10 < angle < 10:
                    if y1 < h / 5 and y2 < h / 5:
                        if y1 == 0:
                            top_lines.append((x1, y1 + 1, x2, y2))
                        else:
                            top_lines.append((x1, y1, x2, y2))
                        # cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    elif y1 > h / 1.5 and y2 > h / 1.5:
                        # else:
                        bottom_lines.append((x1, y1, x2, y2))
                        # cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                elif 80 < angle < 100 or -100 < angle < -80:
                    if x1 < w / 2:
                        left_lines.append((x1, y1, x2, y2))
                    elif x1 > w / 2:
                        right_lines.append((x1, y1, x2, y2))
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    def sortFirst(val):
        return val[0]

    # function to return the second element of the
    # two elements passed as the paramater
    def sortSecond(val):
        return val[1]

    # sorts the array in ascending according to second element
    if len(top_lines) > 1:
        top_lines.sort(key=sortSecond, reverse=True)

    # sorts the array in ascending according to second element
    if len(bottom_lines) > 1:
        bottom_lines.sort(key=sortSecond)

    if len(left_lines) > 1:
        left_lines.sort(key=sortFirst, reverse=True)

    if len(right_lines) > 1:
        right_lines.sort(key=sortFirst)

    cropped_plate = invert_input_image[top_lines[0][1]:bottom_lines[0][1], left_lines[0][0]:right_lines[0][0]]
    # plt.imshow(binary_plate2, cmap="gray")
    # plt.show()
    # cv2.imshow('binary_plate2',canny)

    cropped_plate = cv2.bitwise_not(cropped_plate)

    #
    # binary_plate2 = cv2.copyMakeBorder(binary_plate2, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)
    cropped_plate = cv2.copyMakeBorder(cropped_plate, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    # crop_img = cv2.copyMakeBorder(crop_img, 1,1,1,1, cv2.BORDER_CONSTANT, None, (255,255,255))
    # binary_plate2 = np.uint8(cropped_plate)

    return cropped_plate

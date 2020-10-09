import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


def cropp_by_hough_lines(input_img):
    input_img = cv2.resize(input_img, (382, 84))
    invert_input_img = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    # ret, invert_input_img = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h = invert_input_img.shape[0]
    w = invert_input_img.shape[1]

    kernel_horizontal = np.ones((1, 13), np.uint8)

    img_close_horizontal = cv2.morphologyEx(invert_input_img, cv2.MORPH_CLOSE, kernel_horizontal)

    canny_img_horizontal = cv2.Canny(img_close_horizontal, 0, 200)

    left_lines = [(0, 0, 0, h)]
    right_lines = [(w, 0, w, h)]
    top_lines = [(0, 0, w, 0)]
    bottom_lines = [(0, h, w, h)]

    # minLineLength - minimum length of line
    # w - image width
    # maxLineGap - maximum allowed gap between line segments to treat them as single line
    minLineLength = w * 0.7
    maxLineGap = 4
    lines = cv2.HoughLinesP(canny_img_horizontal, 1, np.pi / 180, 20, minLineLength, maxLineGap)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if y2 - y1 < 10:
                    if y1 < h / 5 and y2 < h / 5:
                        if y1 == 0:
                            top_lines.append((x1, y1 + 1, x2, y2))
                        else:
                            top_lines.append((x1, y1, x2, y2))
                    elif y1 > h * 0.8 and y2 > h * 0.8:
                        bottom_lines.append((x1, y1, x2, y2))

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

    cropped_plate = invert_input_img[top_lines[0][1] : bottom_lines[0][1], 0:w]
    cropped_input_img = input_img[top_lines[0][1] : bottom_lines[0][1], 0:w]

    h2 = cropped_plate.shape[0]
    w2 = cropped_plate.shape[1]

    kernel_vertical = np.ones((13, 3), np.uint8)
    img_close_vertical = cv2.morphologyEx(cropped_plate, cv2.MORPH_CLOSE, kernel_vertical)
    canny_img_vertical = cv2.Canny(img_close_vertical, 0, 200)

    minLineLength2 = h2 * 0.8
    maxLineGap2 = 4
    lines2 = cv2.HoughLinesP(canny_img_vertical, 1, np.pi / 180, 20, minLineLength2, maxLineGap2)
    if lines2 is not None:
        for line in lines2:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if angle == 90 or angle == -90:
                    if x1 < w / 7 and x2 < w / 7:
                        left_lines.append((x1, y1, x2, y2))
                    elif x1 > w * 0.8 and x2 > w * 0.8:
                        right_lines.append((x1, y1, x2, y2))


    if len(left_lines) > 1:
        left_lines.sort(key=sortFirst, reverse=True)
        # while (left_lines[0][0] - left_lines[1][0] < 4) or (left_lines[0][1] - left_lines[0][3] < h2 / 3.2):
        #     del left_lines[0]
        #     if len(left_lines) == 1:
        #         break

    if len(right_lines) > 1:
        right_lines.sort(key=sortFirst)

        while (right_lines[1][0] - right_lines[0][0] > 5) or (right_lines[0][1] - right_lines[0][3] < h2 / 2):
            del right_lines[0]
            if len(right_lines) == 1:
                break


    cropped_plate = cropped_plate[0: h, left_lines[0][0]:right_lines[0][0]]
    cropped_input_img = cropped_input_img[0: h, left_lines[0][0]:right_lines[0][0]]
    
    # rgb_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    # for x1,y1,x2,y2 in top_lines:
    #     cv2.line(rgb_img,(x1,y1),(x2,y2),(255,0,0),2)
    # for x1,y1,x2,y2 in bottom_lines:
    #     cv2.line(rgb_img,(x1,y1),(x2,y2),(0,255,255),2)
    # for x1,y1,x2,y2 in left_lines:
    #     cv2.line(rgb_img,(x1,y1),(x2,y2),(0,255,0),2)
    # for x1,y1,x2,y2 in right_lines:
    #     cv2.line(rgb_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    # fig, axarr = plt.subplots(2)

    # axarr[0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), cmap="gray");
    # axarr[1].imshow(cropped_input_img, cmap="gray");
    # plt.show()

    cropped_plate = cv2.bitwise_not(cropped_plate)

    cropped_plate = cv2.copyMakeBorder(cropped_plate, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    return cropped_input_img

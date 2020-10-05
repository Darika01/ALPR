import cv2
import sys

import numpy as np
from matplotlib import pyplot as plt

def detectPlate(gray_car_image, detector):
    # print("detector", detector)

    plate_dimensions = (
        # 0.02 * gray_car_image.shape[0], 0.20 * gray_car_image.shape[0],
        # 0.1 * gray_car_image.shape[1], 0.38 * gray_car_image.shape[1])
        0.02 * gray_car_image.shape[0], 0.16 * gray_car_image.shape[0],
        0.1 * gray_car_image.shape[1], 0.28 * gray_car_image.shape[1])

    min_height, max_height, min_width, max_width = plate_dimensions
    
    
    if detector == 'tophat':
        kernel = np.ones((int(min_height / 2), int(min_height * 2)), np.uint8)
        #  --> return the difference between input image and Opening of the image
        tophat = cv2.morphologyEx(gray_car_image, cv2.MORPH_TOPHAT, kernel)

        kernel_horizontal = np.ones((1, int(min_width)), np.uint8)
        close = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, kernel_horizontal)

        kernel_vertical = np.ones((int(min_height), 1), np.uint8)
        img_open_vertical = cv2.morphologyEx(close,
            cv2.MORPH_OPEN, kernel_vertical)

        kernel_horizontal = np.ones((1, int(min_width)), np.uint8)
        open_horizontal = cv2.morphologyEx(img_open_vertical,
            cv2.MORPH_OPEN, kernel_horizontal)

        ret, outputImage = cv2.threshold(open_horizontal, 0, 255, cv2.THRESH_OTSU)


    elif detector == 'sobel':
        median = cv2.medianBlur(gray_car_image, 3)
        ret, binary_car_image = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # * sobel
        img_sobelx = cv2.Sobel(binary_car_image, cv2.CV_8U, 1, 0, ksize=3)
        img_sobely = cv2.Sobel(binary_car_image, cv2.CV_8U, 0, 1, ksize=3)
        img_edge = img_sobelx + img_sobely

    elif detector == 'prewitt':
        median = cv2.medianBlur(gray_car_image, 3)
        ret, binary_car_image = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        
        # * prewitt
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(binary_car_image, -1, kernelx)
        img_prewitty = cv2.filter2D(binary_car_image, -1, kernely)
        img_edge = img_prewittx + img_prewitty
    
    elif detector == 'canny':
        gauss = cv2.GaussianBlur(gray_car_image, (3, 5), 0)        
        ret, binary_car_image = cv2.threshold(gauss, 0, 255, cv2.THRESH_OTSU)
        # * canny
        img_edge = cv2.Canny(binary_car_image, 250, 255)

    else:
        print("Entered wrong detector")
        sys.exit(1)
    
    if detector == 'sobel' or detector == 'prewitt' or detector == 'canny':
        # --> morphological operations: vertical open and horizontal close
        kernel_vertical = np.ones((int(min_height / 3), 1), np.uint8)
        # kernel_vertical = np.ones((int(min_height // 5), 1), np.uint8)
        img_open_vertical = cv2.morphologyEx(img_edge, cv2.MORPH_OPEN, kernel_vertical)
        
        kernel_horizontal = np.ones((1, int(min_width / 3)), np.uint8)
        outputImage = cv2.morphologyEx(img_open_vertical, cv2.MORPH_CLOSE, kernel_horizontal)

    # cv2.imshow('detection', outputImage)
    # cv2.waitKey(0)

    # --> connected regions
    # this gets all the connected regions and groups them together
    label_image = cv2.connectedComponents(outputImage)[1]
    contours_image, _ = cv2.findContours(outputImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []

    for contour in contours_image:
        area = cv2.minAreaRect(contour)

        center, size, angle = area[0], area[1], area[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        if angle < -45:
            angle += 90
            w = size[0]
            h = size[1]
            size = (h, w)
        
        region_width, region_height = size
        if region_width != 0 and region_height != 0:
            if 3 < region_width / region_height < 6 and -45 < angle < 45:

                if min_height <= region_height <= max_height and \
                        min_width <= region_width <= max_width and \
                        3.2 * region_height < region_width < 9 * region_height:
                    plates.append((center, size, angle))

    return plates

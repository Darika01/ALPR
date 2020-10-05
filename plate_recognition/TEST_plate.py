import random
import time
import os
from matplotlib import pyplot as plt

import cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import pytesseract

# import segm_label2 as segment
import plate_cropp_projection as plate_cropp_proj
import plate_cropp_haaf_lines as plate_cropp_haaf

import plate_cropp_haaf_lines_for_haarPlate as plate_cropp_haaf_haarPlate

import plate_segm_label as segment
import segmentation_projection as segment_projection

dir = os.path.dirname(__file__)
# image_uri = os.path.join(dir, '../images/plate.png')
image_uri = os.path.join(dir, '../images/plate_segm.jpg')
# image_uri = os.path.join(dir, '../images/test_plates_382x84/280.jpg')

image = cv2.imread(image_uri)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def main():
    plates_segmented = []
    plate_cropped = plate_cropp_proj.cropp_by_projection(gray_image)
    # plate_cropped_haaf = plate_cropp_haaf.cropp_by_haaf_lines(gray_image)
    # plate_cropped_haaf_haarPlate = plate_cropp_haaf_haarPlate.cropp_by_haaf_lines(gray_image)
    if plate_cropped is not None:
        # invert_input_img = cv2.bitwise_not(plate_cropped)
        # text7 = pytesseract.image_to_string(plate_cropped, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 7")
        # text11 = pytesseract.image_to_string(plate_cropped, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 11")
        # print('text7', text7)
        # print('tex11', text11)
        
        # cv2.imshow('plate_cropped', plate_cropped)
        # # cv2.imshow('plate_cropped_haaf', plate_cropped_haaf)
        # cv2.imshow('plate_cropped_haaf_haarPlate', plate_cropped_haaf_haarPlate)
        # cv2.waitKey(0)
        rgb_img = cv2.cvtColor(plate_cropped, cv2.COLOR_GRAY2BGR)

        plate_segm = segment.segm_label(plate_cropped)
        # plate_segm_projection, sumCols = segment_projection.segm_label(plate_cropped, rgb_img)
        # plate_segm_haaf_haarPlate = segment.segm_label(plate_cropped_haaf_haarPlate)
        # ima = Image.open(image_uri)
        # font = ImageFont.truetype(r'Helvetica.ttf', 16)
        # # font = ImageFont.truetype(filename="Helvetica.ttf", size=40)
        # draw = ImageDraw.Draw(ima)
        # draw.text((10, 10), text7, fill=(255, 255, 0), font=font)
        # draw.text((10, 40), text11, fill=(255, 255, 0), font=font)
        # ima.save('sample-out.jpg')

        # if plate_segm is not None:
        #     cv2.imshow('plate_segm', plate_segm)
        # else:
        #     cv2.imshow('plate_cropped', plate_cropped)
        # cv2.waitKey(0)

        # fig, arr = plt.subplots(2, 2)

        # arr[0][0].imshow(plate_cropped, cmap="gray")
        # # arr[0][1].imshow(plate_cropped_haaf_haarPlate, cmap="gray")
        # arr[1][0].imshow(plate_segm, cmap="gray")
        # # arr[0][1].imshow(plate_segm_projection, cmap="gray")
        # # arr[1][1].plot(sumCols)

        # arr[0][0].set_title('a)')
        # arr[0][1].set_title('b)')
        # arr[1][0].set_title('c)')
        # arr[1][1].set_title('d)')
        # plt.show()

        # plates_segmented.append(segment.segm_label(plate_cropped))
    # segment_end_time = round(time.process_time() - segment_time, 2)
    # stimes = stimes + segment_end_time

    # aa = str(random.randint(0,500)) + '.jpg'
    # plt.imsave(aa, ad, cmap="gray")

    # if vert_projection.validate_plate(img_crop)[0]:
    #     # --> draw boxes after plates verifying by checking vertical projection
    #     box = cv2.boxPoints(area)
    #     box = np.int0(box)
    #     cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    #     # cv2.imshow('dd', img_crop)
    #     # aa = str(x) + '.jpg'
    #     # plt.imsave(aa, img_crop, cmap="gray")
    #     segment_time = time.process_time()
    #     segment.segm_label(ima)
    #     segment_end_time = round(time.process_time() - segment_time, 2)
    #     stimes = stimes + segment_end_time

#     prev = area
# else:
#     prev = area



main()

import random
import time
import os
import glob
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
# image_uri = os.path.join(dir, '../images/test_plates_v2/280.jpg')

# images = [cv2.imread(file) for file in glob.glob("C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/images/test_plates_v2/*.jpg")]
path = glob.glob("C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/images/test_plates_v2/*.jpg")
result_path = "C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/plate_recognition/tesseract_result/"

image = cv2.imread(image_uri)

def main(img_path, img, result_path_name):
    plates_segmented = []
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plate_cropped = plate_cropp_proj.cropp_by_projection(gray_image)
    # plate_cropped_haaf = plate_cropp_haaf.cropp_by_haaf_lines(gray_image)
    # plate_cropped_haaf_haarPlate = plate_cropp_haaf_haarPlate.cropp_by_haaf_lines(gray_image)
    if plate_cropped is not None:
        # invert_input_img = cv2.bitwise_not(plate_cropped)
        text7 = pytesseract.image_to_string(plate_cropped, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -l eng --oem 3 --psm 7")
        text11 = pytesseract.image_to_string(plate_cropped, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -l eng --oem 3 --psm 11")
        if 6 < len(text7) < 11:
            text7 = text7
        else:
            text7 = 'No plate'
        if 6 < len(text11) < 11:
            text11 = text11
        else:
            text11 = 'No plate'

        # print('text7', text7)
        # print('tex11', text11)

        rgb_img = cv2.cvtColor(plate_cropped, cv2.COLOR_GRAY2BGR)

        ima = Image.open(img_path)
        font = ImageFont.truetype(r'Helvetica.ttf', 16)
        draw = ImageDraw.Draw(ima)
        draw.text((10, 10), text7, fill=(255, 255, 0), font=font)
        draw.text((10, 40), text11, fill=(255, 255, 0), font=font)
        ima.save(result_path_name)


for img_path in path:
    img = cv2.imread(img_path)
    # print(result_path, img, i.shape[1], i.shape[0])
    result_path_name = img_path.replace("C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/images/test_plates_v2\\", result_path)
    # print(img_path, result_path_name)
    main(img_path, img, result_path_name)
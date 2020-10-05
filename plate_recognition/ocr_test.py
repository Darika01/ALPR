import random
import time
import os
import argparse
import sys
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import plate_cropp_projection as plate_cropp_proj

path = glob.glob("C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/plates_new/*.png")
mainPath = 'C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/plates_new/'

dir = os.path.dirname(__file__)
# image_uri = os.path.join(dir, '../images/plate_cropped.png')
def segm_label(input_img, path, outputFile):
    result = [];
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    contours_image, _ = cv2.findContours(np.uint8(input_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    character_dimensions = (0.5 * input_img.shape[0], 0.99 * input_img.shape[0],
                            0.025 * input_img.shape[1], 0.18 * input_img.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter = 0
    column_list = []
    letters = []
    prevx = 0
    
    for contour in contours_image:
        # y0, x0, y1, x1 = regions.bbox
        x, y, region_width, region_height = cv2.boundingRect(contour)
        center = x + region_width / 2, y + region_height / 2

        if min_height < region_height < max_height and min_width < region_width < max_width:
            centerx = x + region_width / 2
            if abs(centerx - prevx) > region_width / 2:
                letters.append((x, y, region_width, region_height))
                prevx = centerx
                # return prevx
            # else:
            #     prevx = centerx

    if 5 < len(letters) < 11:
        letters = sorted(letters, key=lambda tup: tup[0])
        print('letters', letters)
        for letter in letters:
            x, y, region_width, region_height = letter
            # if x > 2:
            if (x + region_width > input_img.shape[1] - 2) and (region_width < 5):
                continue
            invert_input_img = cv2.bitwise_not(input_img)
            roi = invert_input_img[y: y + region_height, x: x + region_width]
            
            resized = cv2.resize(roi, (30 * region_width // region_height, 30))
            # draw a red bordered rectangle over the character
            cv2.rectangle(rgb_img, (x, y), (x + region_width, y + region_height), (255, 0, 0), 1)
            # resize the characters and then append each character into the characters list
            resized_letters = resized.copy()
            (h, w) = resized_letters.shape[:2]
            characters.append(resized_letters)
            # cv2.imshow('ima', resized_letters)
            # cv2.waitKey(0)
            filename = mainPath+'result/img.jpg'
            plt.imsave(filename, resized_letters, cmap="gray")
            img = Image.new('L', (int(w*2), int(h*2)), color = 'white')
            im = Image.open(mainPath+"result/img.jpg")
            character_back_img = img.copy()
            character_back_img.paste(im, (int(w//2), int(h//2)))
            img_name = mainPath+'result/' + str(x) + str(y) + str(random.randint(0,1000)) + '.jpg'
            plt.imsave(img_name, character_back_img, cmap="gray")
            text = pytesseract.image_to_string(character_back_img,lang='eng', config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10")
            os.remove(filename)
            os.remove(img_name)
            print(text)
            result.append(text)
            image_name = str(random.randint(0,500)) + '.jpg'
            # plt.imsave(image_name, character_back_img, cmap="gray")

            # keep track of the arrangement of the characters
            column_list.append(x)

        res = ''.join(result)
        print('result: ', res)
        ima = Image.open(path)
        font = ImageFont.truetype(r'C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/plates_new/Helvetica.ttf', 16)
        draw = ImageDraw.Draw(ima)
        draw.text((10, 10), res, fill=(255, 0, 0), font=font)
        ima.save(outputFile)
        return rgb_img

outputFile = "detect_plate_output.txt"
# resultTextFile = open('C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/plates_new/result.txt', 'w')
print(len('C:/Users/dkukareka/Desktop/dk/mgr/finished_algorithms/plates_new/'), path[0][65:-4])

for img in path:
    cap = cv2.imread(img)
    outputFile = mainPath + 'result/' + img[65:-4] + '_recognition_output.jpg'
   
    cap = cv2.imread(img)

    resized_image = cv2.resize(cap, (382, 84))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # ret, outputImage = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    plate_cropped = plate_cropp_proj.cropp_by_projection(gray_image)
    # cv2.imshow('ima', plate_cropped)
    # cv2.waitKey(0)

    segm_label(plate_cropped, img, outputFile)

# invert_input_img = cv2.bitwise_not(plate_cropped)
# text7 = pytesseract.image_to_string(invert_input_img, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 7")
# text11 = pytesseract.image_to_string(invert_input_img, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 11")
# print('text7', text7)
# print('tex11', text11)

import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pytesseract
import os
import random

def segm_label(input_img):
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

    if 6 < len(letters) < 11:
        letters = sorted(letters, key=lambda tup: tup[0])
        print('letters', letters)
        for letter in letters:
            x, y, region_width, region_height = letter
            # if x > 2:
            if (x + region_width > input_img.shape[1] - 2) and (region_width < 5):
                continue
            invert_input_img = cv2.bitwise_not(input_img)
            roi = invert_input_img[y:y + region_height, x:x + region_width]

            # draw a red bordered rectangle over the character
            cv2.rectangle(rgb_img, (x, y), (x + region_width, y + region_height), (255, 0, 0), 1)
            # resize the characters and then append each character into the characters list
            # resized_letters = resize(roi, (22, 16))
            # resized_letters = resize(roi, (42, 30))
            # resized_letters = resize(roi, (120, 90))
            resized_letters = roi.copy()
            (h, w) = resized_letters.shape[:2]
            characters.append(resized_letters)
            # filename = 'letters/img.jpg'
            # plt.imsave(filename, resized_letters, cmap="gray")
            # img = Image.new('L', (int(w*1.4), int(h*1.2)), color = 'white')
            # im = Image.open("letters/img.jpg")
            # character_back_img = img.copy()
            # character_back_img.paste(im, (int(w//5), int(h//9)))
            # img_name = 'letters/' + str(x) + str(y) + str(random.randint(0,1000)) + '.jpg'
            # # plt.imsave(img_name, character_back_img, cmap="gray")
            # text = pytesseract.image_to_string(character_back_img, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10")
            # os.remove(filename)
            # print(text)
            # image_name = str(random.randint(0,500)) + '.jpg'
            # plt.imsave(image_name, resized_letters, cmap="gray")

            # keep track of the arrangement of the characters
            column_list.append(x)

        # fig, ayarr = plt.subplots(len(characters), 1, squeeze=False)
        # print('characters', characters)
        # column_list = np.array(column_list)
        # resized_letters = np.array(resized_letters)
        # inds = resized_letters.argsort()
        # sorted_resized_letters = column_list[inds]
        print('col', column_list, resized_letters)


        # # --> show characters
        # for i in characters:
        #     print('character_id', characters.index(i))
        #     # area = plates[i]
        #     # box = cv2.boxPoints(area)
        #     # box = np.int0(box)
        #     # cv2.drawContours(car_image, [box], 0, (0, 0, 255), 2)

        #     # final_plate = plate_like_objects[i]

        #     # print(len(license_plate_id))

        #     ayarr[characters.index(i)].imshow(i, cmap="gray")
            # aa = str(license_plate_id.index(i)) + '.jpg'
            # # plt.imsave(aa, final_plate, cmap="gray")


        # plt.subplot(122), plt.imshow(closing, cmap="gray")
        # # plt.subplot(221), plt.imshow(binary_car_image, cmap="gray")

        # plt.show()

        return rgb_img

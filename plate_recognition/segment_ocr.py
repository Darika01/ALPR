import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image

def segmentAndRecognize(input_img, dirname, base):
    predicted_result_segment = [];
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    # thresh_img = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh_img = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6)
    contours_image, _ = cv2.findContours(np.uint8(thresh_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    character_dimensions = (0.5 * thresh_img.shape[0], 1 * thresh_img.shape[0],
                            0.0025 * thresh_img.shape[1], 0.38 * thresh_img.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    letters = []
    prevx = 0
    
    for contour in contours_image:
        x, y, region_width, region_height = cv2.boundingRect(contour)
        center = x + region_width / 2, y + region_height / 2

        if min_height < region_height < max_height and min_width < region_width < max_width:
            centerx = x + region_width / 2
            if abs(centerx - prevx) > region_width / 2:
                letters.append((x, y, region_width, region_height))
                prevx = centerx
    if 5 <= len(letters) <= 8:
        letters = sorted(letters, key=lambda tup: tup[0])
        for letter in letters:
            x, y, region_width, region_height = letter
            if (x + region_width > thresh_img.shape[1] - 2) and (region_width < 5):
                continue
            invert_thresh_img = cv2.bitwise_not(thresh_img)
            letter_img = thresh_img[y: y + region_height, x: x + region_width]
            
            # draw a red bordered rectangle over the character
            cv2.rectangle(rgb_img, (x, y), (x + region_width, y + region_height), (255, 0, 0), 1)

            # resize the characters and then append each character into the characters list
            resized_letter = cv2.resize(letter_img, (20 * region_width // region_height, 20))
            (h, w) = resized_letter.shape[:2]
            
            filename = dirname + '/temp_result.jpg'
            plt.imsave(filename, resized_letter, cmap="gray")

            bgr_img = Image.new('L', (int(w*2), int(h*2)), color = 'white')
            resized_letter_img = Image.open(filename)
            character_bgr_img = bgr_img.copy()
            character_bgr_img.paste(resized_letter_img, (int(w // 2), int(h // 2)))
            
            whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            result_text = pytesseract.image_to_string(character_bgr_img,
                config=f"-c tessedit_char_whitelist={whitelist} -l eng --psm 10")
            os.remove(filename)
            # plt.imsave(dirname + '/result/temp_result_' + str(letters.index(letter)) + '.jpg', character_bgr_img, cmap="gray")
            predicted_result_segment.append(result_text)

    return rgb_img, ''.join(predicted_result_segment)
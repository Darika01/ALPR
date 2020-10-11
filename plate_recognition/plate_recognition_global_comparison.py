import os
import glob
import cv2
import matplotlib.pyplot as plt
import sys
from itertools import zip_longest
import time

import pytesseract

import plate_cropp_projection as plate_cropp_proj
import plate_cropp_hough_lines as plate_cropp_hough

import segment_ocr


dir = os.path.dirname(__file__)
# path = glob.glob(os.path.join(dir, 'license_plates/*'))
path = glob.glob(os.path.join(dir, 'cropped_license_plates/*.jpg'))

list_license_plates = [] 
predicted_license_plates = []
predicted_segm_license_plates = []

def calculate_predicted_accuracy(actual_list, predicted_list, predicted_segm_license_plates):
    accuracy_sum = 0
    accuracy_segm_sum = 0
    correct = 0
    correct_segm = 0
    for actual_plate, predict_plate, predict_plate_segm in zip(actual_list, predicted_list, predicted_segm_license_plates): 
        accuracy = "0 %"
        num_matches = 0
        accuracy_segm = "0 %"
        num_matches_segm = 0

        if actual_plate == predict_plate: 
            accuracy = "100 %"
            accuracy_sum += 100
            correct += 1
        else: 
            if len(actual_plate) == len(predict_plate): 
                for a, p in zip_longest(actual_plate, predict_plate, fillvalue='?'): 
                    if a == p:
                        num_matches += 1
                if num_matches == 0:
                    accuracy = "0 %"
                # else:
                #     print(num_matches)
                #     if len(actual_plate) != len(predict_plate) and num_matches > 2:
                #         num_matches = num_matches - abs(len(predict_plate) - len(actual_plate))
                accuracy = str(round((num_matches / len(actual_plate)), 2) * 100)
                accuracy_sum += float(accuracy)
                accuracy += "%"

        if actual_plate == predict_plate_segm: 
            accuracy_segm = "100 %"
            accuracy_segm_sum += 100
            correct_segm += 1
        else: 
            if len(actual_plate) == len(predict_plate_segm): 
                for a, p in zip_longest(actual_plate, predict_plate_segm, fillvalue='?'): 
                    if a == p:
                        num_matches_segm += 1
                if num_matches_segm == 0:
                    accuracy_segm = "0 %"
                accuracy_segm = str(round((num_matches_segm / len(actual_plate)), 2) * 100)
                accuracy_segm_sum += float(accuracy_segm)
                accuracy_segm += "%"
        if predict_plate == '':
            predict_plate = '      '
        print("     ", actual_plate, "\t\t\t", predict_plate, "\t\t  ", accuracy, "\t\t\t", predict_plate_segm, "\t\t  ", accuracy_segm)

    return accuracy_sum, accuracy_segm_sum, correct, correct_segm

lp_time = 0
lp_time_segm = 0
print('Processing...')
for img_path in path:
    base = os.path.basename(img_path)
    dirname = os.path.dirname(img_path)
    license_plate_file = os.path.splitext(base)[0]
    ''' 
    Here we append the actual license plate to a list 
    '''
    list_license_plates.append(license_plate_file)
    
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plate_cropped = plate_cropp_proj.cropp_by_projection(gray_img)
    # plate_cropped_hough = plate_cropp_hough.cropp_by_hough_lines(gray_img)
    
    whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    start_time = time.process_time()
    predicted_result = pytesseract.image_to_string(gray_img,
        config=f"-c tessedit_char_whitelist={whitelist} -l eng --psm 7")
    process_time = round((time.process_time() - start_time) / len(base), 5)
    lp_time = lp_time + process_time

    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    predicted_license_plates.append(filter_predicted_result)

    start_time_segm = time.process_time()
    rgb_img, predicted_result_segm = segment_ocr.segmentAndRecognize(gray_img, dirname, base)
    process_time = round((time.process_time() - start_time) / len(base), 5)
    lp_time_segm = lp_time_segm + process_time

    # fig, arr = plt.subplots(3)

    # arr[0].imshow(img, cmap="gray")
    # arr[1].imshow(gray_img, cmap="gray")
    # arr[2].imshow(rgb_img, cmap="gray")

    # arr[0].set_title('Input plate')
    # arr[1].set_title('Analized image')
    # arr[2].set_title('Result segmented image')
    # plt.show()

    predicted_segm_license_plates.append(predicted_result_segm)


print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy", "\t", "Predicted Segmented License Plate", "\t", "Segmented Accuracy") 
print("--------------------", "\t", "-----------------------", "\t", "--------", "\t", "---------------------------------", "\t", "------------------")

accuracy_sum, accuracy_sum_segm, correct, correct_segm = calculate_predicted_accuracy(list_license_plates, predicted_license_plates, predicted_segm_license_plates)


print("-------------------------------------------")
print("\n\nRESULTS\n")
print("-------------------------------------------")

print('Single character recognition process time: ', round(lp_time/len(path)*1000, 2), ' ms')
print('Single character recognition process time (segmened): ', round(lp_time_segm/len(path)*1000, 2), ' ms')

print("Images count", "\t", "Full LP correctly recognized", "\t", "Full LP correctly recognized (segmented LP)", "\t", "Average accuracy", "\t", "Average accuracy (segmented LP)") 
print("------------", "\t", "----------------------------", "\t", "-------------------------------------------", "\t", "----------------", "\t", "-------------------------------")

print("    ", len(path), "\t\t\t", correct, "\t\t\t\t", correct_segm, "\t\t\t\t", round(accuracy_sum / len(path), 2), '%', "\t\t  ", round(accuracy_sum_segm / len(path), 2), '%')

import os
import cv2
import matplotlib.pyplot as plt
import argparse
import sys

import pytesseract

import plate_cropp_projection as plate_cropp_proj
import plate_cropp_hough_lines as plate_cropp_hough

import segment_ocr
import calculate_predicted_accuracy as calc_pred


dir = os.path.dirname(__file__)
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('-i', '--image', help='Path to image file.')
parser.add_argument('-c', '--crop', default='none', help='Crop algorithm.')
parser.add_argument('-p', '--preprocess', default='none', help="Preprocess algorithm. 'none, thresh, blur'")
parser.add_argument('-s', '--isShowCroppedPlate', action='store_true', help='Showing cropped plate.')
args = parser.parse_args()


if (args.image):
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    base = os.path.basename(args.image)
    dirname = os.path.dirname(args.image)
    license_plate_file = os.path.splitext(base)[0]
    img = cv2.imread(args.image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.crop == 'projection':
        plate_img = plate_cropp_proj.cropp_by_projection(gray_img)
    elif args.crop == 'hough':
        plate_img = plate_cropp_hough.cropp_by_hough_lines(gray_img)
    elif args.crop == 'none':
        plate_img = gray_img
    else:
        print("Entered wrong crop algorithm")
        sys.exit(1)
    
    if args.preprocess == 'blur':
        plate_normilized = cv2.medianBlur(plate_img, 3)
    else:
        plate_normilized = plate_img

    rgb_img, predicted_result_segm = segment_ocr.segmentAndRecognize(plate_normilized, dirname, base)

    if args.isShowCroppedPlate:
        fig, arr = plt.subplots(3)

        arr[0].imshow(img, cmap="gray")
        arr[1].imshow(plate_normilized, cmap="gray")
        arr[2].imshow(rgb_img, cmap="gray")

        arr[0].set_title('Input plate')
        arr[1].set_title('Analized image')
        arr[2].set_title('Result segmented image')
        plt.show()

    print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy") 
    print("--------------------", "\t", "-----------------------", "\t", "--------")
    calc_pred.calculate_predicted_accuracy(license_plate_file, predicted_result_segm)

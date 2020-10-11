import time
import os
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# import verify_plate_by_vertical_projection as vert_projection
from topHat_sobel_prewitt_canny import detectPlate

dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('-i', '--image', help='Path to image file.')
parser.add_argument('-v', '--video', help='Path to video file.')
parser.add_argument('-d', '--detector', help="Detector type. 'tophat, sobel, prewitt, canny'")
args = parser.parse_args()

def mainVideo(cap, outputFile):
    # --> Trained XML classifiers describes some features of some object we want to detect
    car_cascade = cv2.CascadeClassifier(os.path.join(dir, 'cars_haar_cascade.xml'))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoOut = cv2.VideoWriter(outputFile, fourcc, 25, (1280, 720), 255)

    frames_count = 0
    ntimes_frame = 0

    ncars = 0
    nplates = 0
    ntimes = 0
    # --> Loop runs if capturing has been initialized.
    while True:
        # --> Reads frames from a video
        hasFrame, frame = cap.read()
        start_frame_time = time.process_time()

        if hasFrame:
            # --> Convert to gray scale of each frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --> Detects cars of different sizes in the input image
            cars = car_cascade.detectMultiScale(gray, 1.1, 1,
                                                minSize=(int(0.2 * frame.shape[0]), int(0.2 * frame.shape[1])))

            if len(cars) != 0:
                frames_count = frames_count + 1

            for (x, y, w, h) in cars:
                # --> To draw a rectangle in each cars
                # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                car_image = frame[y:y + h, x:x + w]
                gray_car_image = gray[y:y + h, x:x + w]

                start_time = time.process_time()

                plates = detectPlate(gray_car_image, args.detector)

                # --> Check if one plate is not inside another
                prev = [(0, 0), (0, 0), 0]
                for area in plates:
                    center, size, angle = area[0], area[1], area[2]
                    center, size = tuple(
                        map(int, center)), tuple(map(int, size))
                    # --> calculate the rotation matrix
                    M = cv2.getRotationMatrix2D(center, angle, 1)
                    # --> perform the actual rotation and hasFrameurn the image
                    img_rot = cv2.warpAffine(
                        gray_car_image, M, gray_car_image.shape[1::-1], flags=cv2.INTER_LINEAR)
                    # --> now rotated rectangle becomes vertical and we crop it
                    img_crop = cv2.getRectSubPix(
                        img_rot, (size[0], size[1]+4), center)
    
                    plate = cv2.resize(img_crop, (382, 84))
                    # ima = vert_projection.validate_plate(plate)

                    if plate is not None:
                        box = cv2.boxPoints(area)
                        box = np.int0(box)
                        cv2.drawContours(car_image, [box], 0, (0, 0, 255), 2)
                        cv2.imshow('plate', plate)
                        # cv2.waitKey(0)

                if len(plates) != 0:
                    ncars = ncars + 1
                    end_time = round(time.process_time() - start_time, 2)
                    ntimes = ntimes + end_time
            videoOut.write(frame)
            # --> Display frames in a window
            if len(cars) != 0:
                end_frame_time = round(
                    time.process_time() - start_frame_time, 2)
                ntimes_frame = ntimes_frame + end_frame_time

            cv2.imshow('Detekcja samochodowych tablic rejestracyjnych', frame)

            # --> Wait for Esc key to stop
            if cv2.waitKey(33) == 27:
                break

        else:
            break

    cap.release()
    # videoOut.release()
    print("--- %s frames ---" % frames_count)
    print("--- %s ms localiation per frame with cars ---" %
          round((ntimes_frame / frames_count)*1000, 1))
    print("--- %s ms localiation per car ---" %
          round((ntimes / ncars)*1000, 1))

    # --> De-allocate any associated memory usage
    cv2.destroyAllWindows()


def mainImage(car_image, outputFile):
    gray_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

    start_time = time.process_time()

    plates = detectPlate(gray_car_image, args.detector)

    # --> Check if one plate is not inside another
    prev = [(0, 0), (0, 0), 0]
    for area in plates:
        center, size, angle = area[0], area[1], area[2]
        center, size = tuple(
            map(int, center)), tuple(map(int, size))
        # --> calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # --> perform the actual rotation and hasFrameurn the image
        img_rot = cv2.warpAffine(
            gray_car_image, M, gray_car_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        # --> now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(
            img_rot, (size[0], size[1]+4), center)

        plate = cv2.resize(img_crop, (382, 84))
        # verified_plate = vert_projection.validate_plate(plate)

        if plate is not None:
            box = cv2.boxPoints(area)
            box = np.int0(box)
            cv2.drawContours(car_image, [box], 0, (0, 0, 255), 2)
            cv2.imshow('plate', plate)
            # cv2.waitKey(0)
            cv2.imwrite(outputFile, car_image)
            
    end_time = round(time.process_time() - start_time, 2)
    print("--- %s ms localiation per car ---" %
        round(end_time*1000, 1))

outputFile = "detect_plate_output.mp4"
if (args.detector):
    if (args.image):
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv2.imread(args.image)
        outputFile = args.image[:-4]+ '_' + args.detector + '_detect_plate_output.jpg'
        mainImage(cap, outputFile)
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        # --> Capture frames from a video
        cap = cv2.VideoCapture(args.video)
        outputFile = args.video[:-4]+ '_' + args.detector + '_detect_plate_output.mp4'
        mainVideo(cap, outputFile)
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
    else:
        # Webcam input
        cap = cv2.VideoCapture(0)
else:
    print("Entered wrong detector")
    sys.exit(1)


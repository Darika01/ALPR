import random
import time
import os
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--detector', help="Detector type. 'haar, lbp'")
args = parser.parse_args()

if args.detector == 'haar':
    plate_cascade = cv2.CascadeClassifier(os.path.join(dir,
    'plates_haar_cascade.xml'))
elif args.detector == 'lbp':
    plate_cascade = cv2.CascadeClassifier(os.path.join(dir,
    'plates_lbp_cascade.xml'))
    # 'cascade_lbp2.xml'))
else:
    print("Entered wrong detector")
    sys.exit(1)


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

                plates = plate_cascade.detectMultiScale(gray_car_image, 1.9, 1,
                            minSize=(int(0.2 * gray_car_image.shape[0]),
                            int(0.4 * gray_car_image.shape[1])))

                for (x, y, w, h) in plates:
                    cv2.rectangle(car_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
    (h, w) = car_image.shape[:2]
    print(h, w)
    car_image_resized = cv2.resize(car_image, (547 * w // h, 547))
    
    gray_car_image = cv2.cvtColor(car_image_resized, cv2.COLOR_BGR2GRAY)

    start_time = time.process_time()

    start_frame_time1 = time.process_time()
    plates = plate_cascade.detectMultiScale(gray_car_image, 1.9, 1,
                                            minSize=(int(0.2 * gray_car_image.shape[0]),
                                                        int(0.4 * gray_car_image.shape[1])))

    print('bbbb', round(time.process_time() - start_frame_time1, 6))
    for (x, y, w, h) in plates:
        cv2.rectangle(car_image_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # print(plates)
    # cv2.imshow('ima', car_image_resized)
    # cv2.waitKey(0)
                
    end_time = round(time.process_time() - start_time, 2)
    print("--- %s ms localiation per car ---" %
        round(end_time * 1000, 1))
        
    cv2.imwrite(outputFile, car_image_resized)


outputFile = "detect_plate_output.mp4"
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


# License Plate Detection and Recognition

This repository contains the author's implementation of different techniques for vehicle license plate detection and recognition in video sequences.

## Dependencies
The program was written with Python 3.8.3 and the following python packages are required:
- opencv: 4.3.0
- matplotlib: 3.2.2
- numpy: 1.18.5
- PIL: 7.2.0
- pytesseract: 0.3.4

## Haar/LBP cascade training with OpenCV (cascade_train_data)
### Train your own cascade
Prepare positive and negative samples according to the [instruction](https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html) (you can use *cascade_train_data/generate_data.py* to generate Good.dat and Bad.dat from your samples) or use existing files from *cascade_train_data/* folder and run the *opencv_traincascade.exe* program with the relevant data from the console:

```bash
opencv_traincascade.exe -data haarcascade\ -vec cascade\samples.vec -bg Bad.dat -numStages 16
-minHitRate 0.995 -maxFalseAlarmRate 0.4 -numPos 260 -numNeg 598-w 120 -h 30 -mode ALL -preca
lcValBufSize 1024 precalcIdxBufSize 1024
```


## License Plate Detection (plate_detection)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.


### License Plate Recognition Using Horizontal and Vertical Projection

Detect LP from an image
````
python plate_detect_projection.py --image test.jpg
````


### License Plate Recognition Based on Edge Detection Algorithm and Morphological Operations

Detect LP from an image
````
python plate_detect_topHat_sobel_prewitt_canny.py --image test.jpg
````

To detect LP from a video
````
python plate_detect_topHat_sobel_prewitt_canny.py --video test.mp4 --detector canny 
````

With possible detectors: 
- **tophat** (localization using morfological operation TopHat)
- **sobel** (localization using Sobel operator for edge detection)
- **prewitt** (localization using Prewitt operator for edge detection)
- **canny** (localization using Canny edge detector)


### License Plate Recognition Using Haar or LBP Cascade Classifier

Detect LP from an image
````
python plate_detect_haar_lbp.py --image test.jpg
````

To detect LP from a video
````
python plate_detect_haar_lbp.py --video test.mp4 --detector haar
````

With possible detectors: 
- **haar** (Haar cascade)
- **lbp** (LBP operator)


### License Plate Recognition Using Convolutional Neural Network (YOLO)
**Before usage download Yolo weights (*lapi.weights*) from this [Link](https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector?select=lapi.weights)**

Detect LP from an image
````
python cnn/plate_detect.py --image test.jpg
````

To detect LP from a video
````
python cnn/plate_detect.py --video test.mp4
````


## License Plate Recognition

## License
[MIT](https://choosealicense.com/licenses/mit/)
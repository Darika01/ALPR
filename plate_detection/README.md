# License Plate Detection

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
python plate_detect_topHat_sobel_prewitt_canny.py --image test.jpg --detector canny
````

To detect LP from a video
````
python plate_detect_topHat_sobel_prewitt_canny.py --video test.mp4 --detector canny 
````
- -i, --image - Path to image file.\
or \
-v, --video - Path to video file.
- -d, --detector - Edge detector or TopHat.
  - **tophat** - localization using morfological operation TopHat;
  - **sobel** - localization using Sobel operator for edge detection;
  - **prewitt** - localization using Prewitt operator for edge detection;
  - **canny** - localization using Canny edge detector;


### License Plate Recognition Using Haar or LBP Cascade Classifier

Detect LP from an image
````
python plate_detect_haar_lbp.py --image test.jpg --detector haar
````

To detect LP from a video
````
python plate_detect_haar_lbp.py --video test.mp4 --detector haar
````
- -i, --image - Path to image file.\
or \
-v, --video - Path to video file.
- -d, --detector - Cascade Classifier.
  - **haar** - Haar cascade;
  - **lbp** - LBP operator;



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
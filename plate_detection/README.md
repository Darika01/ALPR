# License Plate Detection

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.


## License Plate Recognition Using Horizontal and Vertical Projection

Detect LP from an image
````
python plate_detect_projection.py --image test.jpg
````


## License Plate Recognition Based on Edge Detection Algorithm and Morphological Operations

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


## License Plate Recognition Using Haar or LBP Cascade Classifier

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



## License Plate Recognition Using Convolutional Neural Network (YOLO)
**Before usage download Yolo weights (*lapi.weights*) from this [Link](https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector?select=lapi.weights)**

Detect LP from an image
````
python cnn/plate_detect.py --image test.jpg
````

To detect LP from a video
````
python cnn/plate_detect.py --video test.mp4
````
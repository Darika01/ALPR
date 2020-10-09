# License Plate Detection and Recognition

This repository contains the author's implementation of different techniques for vehicle license plate detection, normalization and recognition in image and video sequences.

## Dependencies
The program was written with Python 3.8.3 and the following python packages are required:
- opencv: 4.3.0
- matplotlib: 3.2.2
- numpy: 1.18.5
- PIL: 7.2.0
- pytesseract: 0.3.4
- more-itertools: 8.4.0
- argparse: 1.1


## Implemented algorithms
Vehicle license plate detection using:
  - morfological operation TopHat;
  - Sobel operator for edge detection;
  - Prewitt operator for edge detection;
  - Canny edge detector;
  - Haar cascade;
  - LBP operator;

Cropping license plate number using:
  - horizontal and vertical projection;
  - Hough transform;

License plate recognition using Tesseract:
  - full text recognition;
  - character segmentation and single character recognition;


## License
[MIT](https://choosealicense.com/licenses/mit/)
# License Plate Normalization and Recognition with Tesseract
To determine the recognition accuracy of the license plate, the name of the file containing the recognizable image must contain the plate number (e.g. WX2394N.jpg).

## Full text recognition
````
python plate_recognition/plate_recognition.py -i /cropped_license_plates/WZ2294N.jpg
````

- -i, --image - Path to image file.
- -c, --crop - Crop algorithm.
  - **none** - don't crop (default value);
  - **projection** - crop by projection;
  - **hough** - crop using Hough transform;
- -p, --preprocess - Preprocess algorithm.
  - **none** - don't use any filter (default value);
  - **thresh** - image thresholding using Otsu's method;
  - **blur** - image normalization using median filter;
- -s, --isShowCroppedPlate - Showing cropped plate


## Text segmentation and single character recognition
Character segmentation was realized by finding rectangular contours with appropriate dimensions.

````
python plate_recognition/plate_recognition_segment.py -i /cropped_license_plates/WZ2294N.jpg
````

- -i, --image - Path to image file.
- -c, --crop - Crop algorithm.
  - **none** - don't crop (default value);
  - **projection** - crop by projection;
  - **hough** - crop using Hough transform;
- -p, --preprocess - Preprocess algorithm.
  - **none** - don't use any filter (default value);
  - **blur** - image normalization using median filter;
- -s, --isShowCroppedPlate - Showing cropped plate.


## Algorithms comparison
It will take all .jpg images from 'cropped_license_plates/*.jpg' and compare the result of both algorithms.

````
python plate_recognition/plate_recognition_global_comparison.py
````
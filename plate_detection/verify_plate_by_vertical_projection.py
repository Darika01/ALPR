import cv2
import numpy as np
from skimage.transform import resize

def validate_plate(inputImage):
    """
    validates the candidate plate objects by using the idea
    of vertical projection to calculate the sum of pixels across
    each column and then find the average.
    This method still needs improvement

    """
    resized_candidate = cv2.resize(inputImage, (186, 40))

    each_candidate = inverted_threshold(resized_candidate)
    height, width = each_candidate.shape

    max_average = 24
    min_average = 15

    total_white_pixels = 0
    sumCols = []
    for j in range(width):
        col = each_candidate[0:height, j:j + 1]
        sumCols.append(np.sum(col / 255))
        total_white_pixels = sum(sumCols)

    # min_average & max_average are threshold values
    average = total_white_pixels / width
    if (average >= min_average) & (average < max_average):
        return inputImage


def inverted_threshold(img):
    """
    used to invert the threshold of the candidate regions of the plate
    localization process. The inversion was necessary
    because the license plate area is white dominated which means
    they have a greater gray scale value than the character region

    """

    ret, binary_car_image = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary_car_image

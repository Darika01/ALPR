from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy, copy
import time
import random
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

dir = os.path.dirname(__file__)

def filterImageGaussMedian():
    image_uri = os.path.join(dir, '../images/car.jpg')
    image = cv2.imread(image_uri)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --> median
    median = cv2.medianBlur(gray_image, 5)
    median2 = cv2.medianBlur(gray_image, 11)

    # --> gauss
    gauss = cv2.GaussianBlur(gray_image, (5, 5), 0)
    gauss2 = cv2.GaussianBlur(gray_image, (11, 11), 0)

    # --> average
    blur = cv2.blur(gray_image,(5,5))
    
    fig, arr = plt.subplots(2, 3)

    arr[0][0].imshow(gray_image, cmap="gray")
    arr[0][1].imshow(median, cmap="gray")
    arr[0][2].imshow(median2, cmap="gray")
    # arr[1][0].imshow(gauss, cmap="gray")
    arr[1][1].imshow(gauss, cmap="gray")
    arr[1][2].imshow(gauss2, cmap="gray")

    arr[0][0].set_title('a)')
    arr[0][1].set_title('b)')
    arr[0][2].set_title('c)')
    arr[1][1].set_title("d)")
    arr[1][2].set_title('e)')
    plt.show()


def salt_pepper_noise(image, prob, noise):
    '''
    Adding salt and pepper noise to image
    prob: Probability of the noise
    noise: 0 - salt, 1 - pepper
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()

            if noise is 0:
                if rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
            else:
                if rdn < prob:
                    output[i][j] = 0
                else:
                    output[i][j] = image[i][j]
    return output


def filterImageErosionDilation():
    image_uri = os.path.join(dir, '../images/car.jpg')
    image = cv2.imread(image_uri)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    salt_noise_image = salt_pepper_noise(gray_image, 0.09, 0)
    pepper_noise_image = salt_pepper_noise(gray_image, 0.09, 1)

    kernel = np.ones((3, 3), np.uint8)
    
    # --> erosion
    erosion_salt = cv2.erode(salt_noise_image, kernel, iterations=1)
    
    # --> dilation
    dilation_pepper = cv2.dilate(pepper_noise_image, kernel, iterations=1)

    # --> bad example
    bad_erosion_pepper = cv2.erode(pepper_noise_image, kernel, iterations=1)

    
    fig, arr = plt.subplots(2, 3)

    arr[0][0].imshow(gray_image, cmap="gray")
    arr[0][1].imshow(salt_noise_image, cmap="gray")
    arr[0][2].imshow(erosion_salt, cmap="gray")
    arr[1][0].imshow(pepper_noise_image, cmap="gray")
    arr[1][1].imshow(dilation_pepper, cmap="gray")
    arr[1][2].imshow(bad_erosion_pepper, cmap="gray")

    arr[0][0].set_title('a)')
    arr[0][1].set_title('b)')
    arr[0][2].set_title('c)')
    arr[1][0].set_title("d)")
    arr[1][1].set_title("e)")
    arr[1][2].set_title("f)")
    plt.show()



def image_open_close():
    image_uri = os.path.join(dir, '../images/tor_binarize.png')
    image = cv2.imread(image_uri)
    
    kernel = np.ones((3, 3), np.uint8)
    image_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    fig, arr = plt.subplots(2, 2)

    arr[0][0].imshow(image, cmap="gray")
    arr[0][1].imshow(image_open, cmap="gray")
    arr[1][1].imshow(image_close, cmap="gray")

    arr[0][0].set_title('a)')
    arr[0][1].set_title('b)')
    arr[1][1].set_title('c)')
    plt.show()


filterImageGaussMedian()
# filterImageErosionDilation()
# image_open_close()
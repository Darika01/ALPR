import cv2
import numpy as np
import os
import glob


dir = os.path.dirname(__file__)

def generateBadData():
    path = glob.glob(os.path.join(dir, 'Bad/*'))
    outputfile = open(os.path.join(dir, 'Bad.dat'), 'w')
    i=0
    for img_url in path:
        x = img_url.replace(os.path.join(dir, 'Bad\\'), 'Bad/')
        outputfile.write(x + '\n')
        i = i + 1
    outputfile.close()
    print("Ilosc probek:", i)

def generateGoodData():
    path = glob.glob(os.path.join(dir, 'Good/*'))
    outputfile = open(os.path.join(dir, 'Good.dat'), 'w')
    i=0
    for img_url in path:
        img = cv2.imread(img_url)
        conc = img_url + ' 1 0 0 ' + str(img.shape[1]) + ' ' + str(img.shape[0])
        x = conc.replace(os.path.join(dir, 'Good\\'), 'Good/')
        outputfile.write(x + '\n')
        i = i + 1
    outputfile.close()
    print("Ilosc probek:", i)

# generateBadData()
generateGoodData()



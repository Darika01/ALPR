from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import sys

dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
args = parser.parse_args()

def plateDetect(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    median = cv2.GaussianBlur(gray_image, (3, 9), 0)
    (height, width) = median.shape[:2]

    plate_dimensions = (
        0.02 * gray_image.shape[0], 0.20 * gray_image.shape[0],
        0.1 * gray_image.shape[1], 0.38 * gray_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions

    canny_image = cv2.Canny(median, 250, 255)

    kernel = np.ones((int(min_height // 3), 1), np.uint8)
    img_open = cv2.morphologyEx(canny_image, cv2.MORPH_OPEN, kernel)
    (h, w) = img_open.shape[:2]
    sumCols = []
    for j in range(w):
        col = 255 - img_open[0:h, j:j + 1]
        sumCols.append(w - np.sum(col / 255))

    "Return a list containing the sum of the pixels in each row"
    sumRows = []
    for j in range(h):
        row = 255 - img_open[j:j + 1, 0:w]  # y1:y2, x1:x2
        sumRows.append(h - np.sum(row / 255))


    def setup_axes(fig, rect, rotation, axisScale, axisLimits, doShift):
        tr_rot = Affine2D().scale(axisScale[0], axisScale[1]).rotate_deg(rotation)

        # This seems to do nothing
        if doShift:
            tr_trn = Affine2D().translate(-90,-5)
        else:
            tr_trn = Affine2D().translate(0,0)

        tr = tr_rot + tr_trn

        grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=axisLimits)

        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
        fig.add_subplot(ax)
        aux_ax = ax.get_aux_axes(tr)

        return ax, aux_ax


    fig = plt.figure(1, figsize=(8, 8))
    # axes = [fig.add_subplot(221),fig.add_subplot(223), fig.add_subplot(222)]
    axes = []
    axisOrientation = [180, 180, 270, 180]
    axisScale = [[1,1],[-0.15,1],[0.05,0.5],[1,1]]
    axisPosition = [221,223,222,224]
    axisLimits = [(0, width, 0, width),
                (0, width, 0, np.max(sumCols)),
                (0, width, 0, np.max(sumRows)),
                (0, width, 0, width)]

    doShift = [False, False, False,False]

    label_axes = []
    for i in range(0, len(axisOrientation)):
        ax, aux_ax = setup_axes(fig, axisPosition[i], axisOrientation[i], 
                                axisScale[i], axisLimits[i], doShift[i])
        axes.append(aux_ax)
        label_axes.append(ax)

    # axes[0].imshow(image[..., ::-1])
    axes[0].imshow(cv2.cvtColor(img_open, cv2.COLOR_GRAY2BGR))
    axes[1].plot(sumCols);
    axes[2].plot(sumRows);
    axes[3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    # label_axes[0].axis["bottom"].label.set_text('Variable 1')
    # label_axes[0].axis["left"].label.set_text('Variable 2')

    for i in range(1,len(label_axes)):
        for axisLoc in ['top','left','right']:
            label_axes[i].axis[axisLoc].set_visible(False)
        label_axes[i].axis['bottom'].toggle(ticklabels=False)    

    fig.subplots_adjust(wspace=0.00, hspace=0.00, left=0.00, right=0.99, top=0.96, bottom=0.0)
    plt.show()

if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.imread(args.image)
    plateDetect(cap)
    print("Done processing !!!")
else:
    print("Provide an image")

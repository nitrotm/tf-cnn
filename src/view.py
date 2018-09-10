###########################################
# view.py
#
# Various image viewing utilities
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, math

import numpy as np

from matplotlib import pyplot as plt


def showgray(img, title="Image", histogram=False):
    """
    Show gray image with histogram (opt).
    """

    scale = 640/max(img.shape)
    scaled = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.suptitle(title, y=0.9)
    plt.axis("off")
    plt.imshow(scaled, cmap='gray', interpolation='bicubic')
    plt.show()

    if histogram:
        x = np.arange(1, 256, 4)
        h = np.reshape(cv2.calcHist([img],[0],None,[64],[0,256]), (64))

        plt.figure()
        plt.suptitle("Histogram: " + title, y=0.9)
        plt.xlim([0,260])
        plt.bar(x, h, width=1.0, color=(0, 0, 0), log=True)
        plt.show()


def showcolor(img, title="Image", histogram=False):
    """
    Show bgr image with histogram (opt).
    """

    scale = 640/max(img.shape)
    scaled = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.suptitle(title, y=0.9)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB), cmap='gray', interpolation='bicubic')
    plt.show()

    if histogram:
        xr = np.arange(1, 256, 4)
        hr = np.reshape(cv2.calcHist([img],[2],None,[64],[0,256]), (64))
        xg = np.arange(2, 256, 4)
        hg = np.reshape(cv2.calcHist([img],[1],None,[64],[0,256]), (64))
        xb = np.arange(3, 256, 4)
        hb = np.reshape(cv2.calcHist([img],[0],None,[64],[0,256]), (64))

        plt.figure()
        plt.suptitle("Histogram: " + title, y=0.9)
        plt.xlim([0,260])
        plt.bar(xr, hr, width=1.0, color=(1, 0, 0), log=True)
        plt.bar(xg, hg, width=1.0, color=(0, 1, 0), log=True)
        plt.bar(xb, hb, width=1.0, color=(0, 0, 1), log=True)
        plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("putin.jpg")
    b,g,r = cv2.split(img)

    img2 = cv2.imread("tramp.jpg")
    b2,g2,r2 = cv2.split(img)

    img3 = cv2.imread("car.jpg")
    b3, g3, r3 = cv2.split(img)
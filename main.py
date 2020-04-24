import cv2
import numpy as np
import matplotlib.pyplot as plt


def B(A, B, k, l):
    rows = A.shape[0]
    cols = A.shape[1]
    rowsA = range(k + 1, rows + 1)
    colsA = range(l + 1, cols + 1)
    rowsB = range(1, rows - k + 1)
    colsB = range(1, cols - l + 1)
    if k < 0:
        k = abs(k)
        rowsA = range(1, rows - k + 1)
        rowsB = range(k + 1, rows + 1)
    if l < 0:
        l = abs(l)
        colsA = range(1, cols - l + 1)
        colsB = range(l + 1, cols + 1)
    return np.sum(np.sum(np.dot(A(rowsA, colsA), B(rowsB, colsB)))) / ((rows - 1) * (cols - 1))


if __name__ == '__main__':
    img = cv2.imread("putin.jpg")
    b, g, r = cv2.split(img)

    img2 = cv2.imread("tramp.jpg")
    b2, g2, r2 = cv2.split(img)

    img3 = cv2.imread("car.jpg")
    b3, g3, r3 = cv2.split(img)

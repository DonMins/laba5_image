import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def addNoise (image,noise_percentage):
    vals = len(image.flatten())
    out = np.copy(image)
    nose = int(np.ceil(noise_percentage * vals / 100))
    # Salt mode
    num_salt = int(nose/2)
    num_pepper = int(nose/2)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1
    # Pepper mode
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out

def getVec_b(img, imgNose):
    b = np.zeros(16)
    i = 0
    for n in range(-1,3):
        for m in range(-1,3):
            b[i] = B(img, imgNose, -n, -m)
            i +=1
    return b


def B(A, B, k, l):
    rows = A.shape[0]
    cols = A.shape[1]
    rowsA = range(k, rows )
    colsA = range(l, cols )
    rowsB = range(0, rows - k )
    colsB = range(0, cols - l )
    if k < 0:
        k = abs(k)
        rowsA = range(0, rows - k )
        rowsB = range(k, rows)
    if l < 0:
        l = abs(l)
        colsA = range(0, cols - l )
        colsB = range(l, cols )

    return np.sum(np.sum(A[rowsA, colsA] * B[rowsB, colsB])) / ((rows - 1) * (cols - 1))



if __name__ == '__main__':
    img = cv2.imread("putin.jpg")
    imgNose = cv2.imread("putinNose.jpg")

    b, g, r = cv2.split(img)

    # imgNose = addNoise(img,13)

    # cv2.imwrite("putinNose.jpg",imgNose)
    bNose, gNose, rNose = cv2.split(imgNose)

    print(getVec_b(img,imgNose))



    img2 = cv2.imread("tramp.jpg")
    b2, g2, r2 = cv2.split(img)

    img3 = cv2.imread("car.jpg")
    b3, g3, r3 = cv2.split(img)


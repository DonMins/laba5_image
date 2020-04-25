import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def addNoise(image, noise_percentage):
    vals = len(image.flatten())
    out = np.copy(image)
    nose = int(np.ceil(noise_percentage * vals / 100))
    # Salt mode
    num_salt = int(nose / 2)
    num_pepper = int(nose / 2)
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
    for n in range(-1, 3):
        for m in range(-1, 3):
            b[i] = B(img, imgNose, -n, -m)
            i += 1
    return b


def matrixA(imgNose):
    matrixA = []
    for n in range(-1, 3):
        for m in range(-1, 3):
            for k in range(-1, 3):
                for l in range(-1, 3):
                    matrixA.append(B(imgNose, imgNose, n - k, m - l))

    matrixA = np.array(matrixA).reshape(16, 16)

    return matrixA


def matrixMask(A, b):
    return np.linalg.solve(A, b)


def B(A, B, k, l):
    rows = A.shape[0]
    cols = A.shape[1]
    rowsA = range(k, rows)
    colsA = range(l, cols)
    rowsB = range(0, rows - k)
    colsB = range(0, cols - l)
    if k < 0:
        k = abs(k)
        rowsA = range(0, rows - k)
        rowsB = range(k, rows)
    if l < 0:
        l = abs(l)
        colsA = range(0, cols - l)
        colsB = range(l, cols)

    return np.sum(np.sum(A[rowsA][colsA] * B[rowsB][colsB])) / ((rows - 1) * (cols - 1))


def Convolution(image, kernel, D):
    heightM, widthM = kernel.shape
    height, width = image.shape

    centerHeightM = D.index(0)
    centerwidthM = D.index(0)

    img2 = np.zeros((height + heightM - 1, width + widthM - 1), np.uint8)
    img2[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM] = image

    for i in range(centerHeightM):
        img2[i, :] = img2[centerHeightM, :]
        img2[height + 1 + i, :] = img2[height, :]

    for i in range(centerwidthM):
        img2[:, i] = img2[:, centerwidthM]
        img2[:, width + 1 + i] = img2[:, width]

    new_image = np.zeros((height + centerHeightM, width + centerwidthM), np.uint8)

    for i in range(centerHeightM, height + centerHeightM):
        for j in range(centerwidthM, width + centerwidthM):
            new_image[i][j] = np.sum( img2[i - centerHeightM: i + (heightM - centerHeightM - 1) + 1,
                j - centerwidthM: j + (widthM - centerwidthM - 1) + 1] * kernel)

    return new_image[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM]


if __name__ == '__main__':
    img = cv2.imread("putin.jpg")
    imgNose = cv2.imread("putinNose.jpg")
    D = [-1, 0, 1, 2]
    # D = [-2, -1, 0, 1]

    b, g, r = cv2.split(img)
    bNose, gNose, rNose = cv2.split(imgNose)
    cv2.imshow("bNose", bNose)

    # imgNose = addNoise(img,13)

    # cv2.imwrite("putinNose.jpg",imgNose)

    bvec = getVec_b(b, bNose)
    A = matrixA(bNose)
    Mask = matrixMask(A, bvec)
    Mask = np.array(Mask).reshape(4, 4)
    print(Mask)

    result = Convolution(bNose, Mask, D)

    cv2.imshow("Approval Method", result)

    cv2.waitKey(0)
    img2 = cv2.imread("tramp.jpg")
    b2, g2, r2 = cv2.split(img)

    img3 = cv2.imread("car.jpg")
    b3, g3, r3 = cv2.split(img)

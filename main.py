import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def addNoise(image, noise_percentage):
    vals = len(image.flatten())
    out = np.copy(image)
    nose = int(np.ceil(noise_percentage * vals / 100))
    # Salt mode
    num_salt = int(nose)
    listAllCoord = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            listAllCoord.append([i, j])

    random.shuffle(listAllCoord)
    for i in range(0, int(num_salt)):
        coord = listAllCoord.pop(np.random.randint(0, len(listAllCoord)))
        nose = np.array(image[coord[0], coord[1]] + [random.normalvariate(0, 20), random.normalvariate(0, 20),
                                                     random.normalvariate(0, 20)], dtype=int)
        for i in range(len(nose)):
            if nose[i] > 255:
                nose[i] = 255
            elif nose[i] < 0:
                nose[i] = 0
        out[coord[0], coord[1]] = nose

    return out


def getVec_b(img, imgNose, D):
    b = np.zeros(16)
    i = 0
    for n in D:
        for m in D:
            b[i] = B(img, imgNose, -n, -m)
            i += 1
    return b


def matrixA(imgNose, D):
    matrixA = []
    for n in D:
        for m in D:
            for k in D:
                for l in D:
                    matrixA.append(B(imgNose, imgNose, n - k, m - l))

    matrixA = np.array(matrixA).reshape(16, 16)

    return matrixA


def matrixMask(A, b):
    return np.linalg.solve(A, b)


def B(A, B, k, l):
    rows = A.shape[0]
    cols = A.shape[1]

    if k < 0 and l < 0:
        return np.sum(np.sum(A[0:rows - abs(k), 0:cols - abs(l)] * B[abs(k):rows, abs(l):cols])) / (
                (rows - 1) * (cols - 1))

    elif k < 0 and l >= 0:
        return np.sum(np.sum(A[0:rows - abs(k), l:cols] * B[abs(k):rows, 0:cols - l])) / ((rows - 1) * (cols - 1))

    elif k >= 0 and l < 0:
        return np.sum(np.sum(A[k:rows, 0:cols - abs(l)] * B[0:rows - k, abs(l):cols])) / ((rows - 1) * (cols - 1))

    return np.sum(np.sum(A[k:rows, l:cols] * B[0:rows - k, 0:cols - l])) / ((rows - 1) * (cols - 1))


def getError(img, img2):
    img3 = np.copy(img)
    img4 = np.copy(img2)
    img3 = img3/255
    img4 = img4/255
    width = img.shape[1]
    height = img.shape[0]
    error = (np.sum((np.array(img3.flatten()) - np.array(img4.flatten())) ** 2,
                    dtype=np.float64) / (width * height * 3)) ** 0.5
    return error


def Convolution(image, kernel, D):
    heightM, widthM = kernel.shape
    height, width = image.shape

    centerHeightM = D.index(0)
    centerwidthM = D.index(0)

    img2 = np.zeros((height + heightM - 1, width + widthM - 1), np.uint8)
    img2[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM] = image

    for i in range(centerHeightM):
        img2[i, :] = img2[centerHeightM, :]

    for i in range(heightM - centerHeightM):
        img2[height + i, :] = img2[height, :]

    for i in range(centerwidthM):
        img2[:, i] = img2[:, centerwidthM]

    for i in range(widthM - centerwidthM):
        img2[:, width + i] = img2[:, width]

    new_image = np.zeros((height + heightM - 1, width + widthM - 1), np.uint8)

    for i in range(centerHeightM, height + 1):
        for j in range(centerwidthM, width + 1):
            pixel = np.sum(img2[i - centerHeightM: i + (heightM - centerHeightM - 1) + 1,
                           j - centerwidthM: j + (widthM - centerwidthM - 1) + 1] * kernel)
            if pixel > 255:
                new_image[i][j] = 255
            elif pixel < 0:
                new_image[i][j] = 0
            else:
                new_image[i][j] = pixel

    return new_image[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM]


if __name__ == '__main__':
    img = cv2.imread("putin.png")
    imgNose = addNoise(img, 13)
    b, g, r = cv2.split(img)
    bNose, gNose, rNose = cv2.split(imgNose)

    D = [-1, 0, 1, 2]

    cv2.imshow("imgNose", imgNose)
    cv2.imshow("img", img)
    cv2.imwrite("imgNose.jpg", imgNose)

    bvec = getVec_b(b, bNose, D)

    A = matrixA(bNose, D)
    Mask = matrixMask(A, bvec)
    Mask = np.array(Mask).reshape(4, 4)
    print(Mask)
    resultB = Convolution(bNose, Mask, D)
    # cv2.imshow("resultB", resultB)

    bvec2 = getVec_b(g, gNose, D)
    A2 = matrixA(gNose, D)
    Mask2 = matrixMask(A2, bvec2)
    Mask2 = np.array(Mask2).reshape(4, 4)
    print(Mask2)

    resultG = Convolution(gNose, Mask2, D)
    # cv2.imshow("resultG", resultG)

    bvec3 = getVec_b(r, rNose, D)
    A3 = matrixA(rNose, D)
    Mask3 = matrixMask(A3, bvec3)
    Mask3 = np.array(Mask3).reshape(4, 4)
    print(Mask3)
    resultR = Convolution(rNose, Mask3, D)
    # cv2.imshow("resultR", resultR)

    m = cv2.merge((resultB, resultG, resultR))
    print(m.shape)

    cv2.imshow("Res", m)
    cv2.imwrite("Res.jpg", m)

    print("Исходная с исходной, ошибка восстановления", getError(img, img))
    print("Исходная с зашумленной, ошибка восстановления", getError(img, imgNose))
    print("Исходная с отфильтрованной ошибка восстановления", getError(img, m))
    print("В процентном соотношении стало лучше на ", 100 - (getError(img, m) * 100 / getError(img, imgNose)), " %")

    imgCar = cv2.imread("car.jpg")
    imgCarNose = addNoise(imgCar, 13)
    cv2.imshow("imgCar", imgCar)
    cv2.imshow("imgCarNose", imgCarNose)
    bCarNose, gCarNose, rCarNose = cv2.split(imgCarNose)
    resultBCar = Convolution(bCarNose, Mask, D)
    resultGCar = Convolution(gCarNose, Mask2, D)
    resultRCar = Convolution(rCarNose, Mask3, D)
    carRes = cv2.merge((resultBCar, resultGCar, resultRCar))

    cv2.imshow("carRes", carRes)
    print("------------------------------------------------------------------")
    print("Исходная с исходной, ошибка восстановления", getError(imgCar, imgCar))
    print("Исходная с зашумленной, ошибка восстановления", getError(imgCar, imgCarNose))
    print("Исходная с отфильтрованной ошибка восстановления", getError(imgCar, carRes))
    print("В процентном соотношении стало лучше на ",
          100 - (getError(imgCar, carRes) * 100 / getError(imgCar, imgCarNose)), " %")

    imgTramp = cv2.imread("tramp.jpg")
    imgTrampNose = addNoise(imgTramp, 13)
    cv2.imshow("imgTramp", imgTramp)
    cv2.imshow("imgTrampNose", imgTrampNose)
    bTrampNose, gTrampNose, rTrampNose = cv2.split(imgTrampNose)
    resultBTramp = Convolution(bTrampNose, Mask, D)
    resultGrTamp = Convolution(gTrampNose, Mask2, D)
    resultRTramp = Convolution(rTrampNose, Mask3, D)
    trampRes = cv2.merge((resultBTramp, resultGrTamp, resultRTramp))

    cv2.imshow("trampRes", trampRes)
    print("------------------------------------------------------------------")
    print("Исходная с исходной, ошибка восстановления", getError(imgTramp, imgTramp))
    print("Исходная с зашумленной, ошибка восстановления", getError(imgTramp, imgTrampNose))
    print("Исходная с отфильтрованной ошибка восстановления", getError(imgTramp, trampRes))
    print("В процентном соотношении стало лучше на ",
          100 - (getError(imgTramp, trampRes) * 100 / getError(imgTramp, imgTrampNose)), " %")

    cv2.waitKey(0)

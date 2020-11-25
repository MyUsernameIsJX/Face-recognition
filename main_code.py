# -*- coding: UTF-8 -*-
# Author: BITU Jiaxing Zhang

import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def loadData(path):
    i = 0
    hang, lie = 50, 40
    files = os.listdir(path)
    data = np.zeros((len(files), hang * lie))
    for file in files:
        filepath = path+'\\'+file
        img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        temp = img.reshape(1, hang*lie)
        data[i, :] = temp[0, :]
        i = i+1
    return data


def my_pca(mat, k):
    average = np.mean(mat, axis=0)
    m, n = mat.shape
    avgs = np.tile(average, (m, 1))
    data_adjust = mat - avgs
    covX = np.cov(data_adjust.T)  # 计算协方差矩阵
    featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue)  # 依照featValue进行从大到小排序

    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])  # 所以这里须要进行转置
        finalData1 = data_adjust * selectVec.T
        # pca = PCA(n_components=k)
        # finalData2 = pca.fit_transform(mat)
        reconData1 = (finalData1 * selectVec) + average
        # reconData2 = (finalData2 * selectVec) + average
    return finalData1, selectVec


def testSetDR(mat, selectVec):
    average = np.mean(mat, axis=0)
    m, n = mat.shape
    avgs = np.tile(average, (m, 1))
    data_adjust = mat - avgs
    finalData = data_adjust * selectVec.T
    # reconData = (finalData * selectVec) + average
    return finalData



def my_pca2(mat, k):
    average = np.mean(mat, axis=0)
    m, n = mat.shape
    avgs = np.tile(average, (m, 1))
    data_adjust = mat - avgs
    covX = np.cov(data_adjust.T)  # 计算协方差矩阵
    featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue)  # 依照featValue进行从大到小排序

    if k > n:
        print("k must lower than feature number")
        return
    else:

        selectVec = np.matrix(featVec.T[index[:k]])
        finalData1 = data_adjust * selectVec.T
    return finalData1, selectVec


def restoreData(mat, hang, lie):
    m, n = mat.shape
    img = []
    for i in range(m):
        img.append(mat[i].reshape(hang, lie).astype(np.uint8))
    return img


def L2_Distance(vector1, vector2):
    return np.linalg.norm(vector1-vector2)


def L1_Distance(vector1, vector2):
    return np.linalg.norm(vector1-vector2, ord=1)


def recognition(featureData, testData, trainNum):
    fdSize = featureData.shape
    tdSize = testData.shape
    foreResult = []
    for i in range(tdSize[0]):
        distances = []
        for j in range(fdSize[0]):
            distances.append(L2_Distance(testData[i, :], featureData[j, :]))
        foreResult.append(int(np.argmin(np.array(distances)) / trainNum))
    fore_result = np.array(foreResult)
    return fore_result


def main(trainNum,featureNum):
    path = '.\AR'
    data_all = loadData(path)
    humanNum = 120
    imgNum = 14
    # testNum = imgNum - trainNum
    testNum = imgNum
    data_train = np.zeros((humanNum*trainNum, data_all.shape[1]))
    # data_test = np.zeros((humanNum*testNum, data_all.shape[1]))
    data_test = data_all

    for i in range(humanNum):
        data_train[i*trainNum:(i+1)*trainNum, :] = data_all[i*imgNum:i*imgNum+trainNum, :]
        # data_test[i*testNum:(i+1)*testNum, :] = data_all[i*imgNum+trainNum:i*imgNum+imgNum, :]

    featureData, selectVec = my_pca2(data_train, featureNum)
    test_Data = testSetDR(data_test, selectVec)
    foreResult = recognition(featureData, test_Data, trainNum).reshape(humanNum, testNum)
    errorNum = 0
    for i in range(humanNum):
        errorNum = errorNum + sum(sum(foreResult[i]!=np.ones((1, testNum))*i))
    errorRate = errorNum/(humanNum*testNum)
    print('trainNum', trainNum, 'featureNum:', featureNum, 'errorRate: ', errorRate)

    return 1-errorRate

    # srcImgs = restoreData(data_train, 50, 40)
    # resImgs1 = restoreData(reconData1, 50, 40)
    # resImgs2 = restoreData(reconData2, 50, 40)
    # img_2 = np.hstack([srcImgs[0], resImgs1[0], resImgs2[0]])
    # cv.imshow('Imgs', img_2)

    # cv.waitKey(0)


if __name__ == "__main__":
    rightRate = np.zeros((10, 15))
    for trainNum in range(10):
        for k in range(15):
            rightRate[trainNum][k] = main(trainNum+1, k*100+100)
            if rightRate[trainNum][k] == rightRate[trainNum][k-1]:
                break
            print('trainNum', trainNum+1, 'featureNum:', k*100+100, 'rightRate: ', rightRate[trainNum][k])
    print(rightRate)
    np.save('result', rightRate)
    print('save done')

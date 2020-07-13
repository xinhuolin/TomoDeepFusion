# -*- coding: utf-8 -*-
# coding=utf-8
from scipy.io import loadmat
import numpy as np
import sys
import struct

def saveData1(path, dataList, name):
    file = open(path + "\\" + "data.dat", "w+")
    for i in range(len(dataList)):  # 1
        dataStr = dataList[i]
        a = dataList[i].real
        b = dataList[i].imag
        c = int(a)
        d = int(b)
        e = hex(c & 0xffff)
        f = hex(d & 0xffff)
        m = e[2:]
        n = f[2:]
        if c >= 0:
            for k in range(4 - len(m)):
                m = "0" + m
        if d >= 0:
            for k in range(4 - len(n)):
                n = "0" + n
        data = m + n
        file.write(data)
        file.write("\n")
    file.close()


def saveData2(path, dataList, name):
    file = open(path + "\\" + "data.dat", "w+")
    for i in range(len(dataList[0])):  # 1
        for j in range(len(dataList)):  # 792
            dataStr = dataList[j][i]
            a = dataList[j][i].real
            b = dataList[j][i].imag
            c = int(a)
            d = int(b)
            e = hex(c & 0xffff)
            f = hex(d & 0xffff)
            m = e[2:]
            n = f[2:]
            if c >= 0:
                for k in range(4 - len(m)):
                    m = "0" + m
            if d >= 0:
                for k in range(4 - len(n)):
                    n = "0" + n
            data = m + n
            file.write(data)
            file.write("\n")
    file.close()


def saveData3(path, dataList, name):
    print(len(dataList[0][0]), len(dataList[0]), len(dataList))
    with open(path + "output_%s.bin"%name, 'wb')as fp:
        for i in range(len(dataList[0][0])):  #
            for j in range(len(dataList[0])):  #
                for k in range(len(dataList)):  #
                    dataStr = struct.pack('f', dataList[k][j][i])
                    fp.write(dataStr)


if __name__ == "__main__":
    # path = sys.argv[1]
    # dataName = sys.argv[2]
    path = "./"
    data_path = "Denoise(49)[wbp].mat"
    print(data_path)
    data = loadmat(data_path)
    print(data.keys())
    for i in data:
        if i == "__version__" or i == "__header__" or i == "__globals__":
            continue
        else:
            dataList = data[i]
            # 把列表转换为numpy的数组格式，再根据shape方法，可以得到mat数据的纬度。
            L = np.array(dataList)
            print(L.shape)
            dataLen = len(L.shape)
            # 一维的时候，L.shape返回<1L,1L>,所以不能根据len的长度判断
            if L.shape[0] == 1 and L.shape[1] == 1:
                saveData1(path, dataList, i)
            elif dataLen == 2:
                saveData2(path, dataList, i)
            else:
                saveData3(path, dataList, i)
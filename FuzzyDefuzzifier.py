#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyDefuzzifier.py
# @Time      :2023/3/27 1:35 PM
# @Author    :Kinddle
import numpy as np


def center_of_sets(Func, xLim=(-1000, 1000), yLim=(0, 1), num=100000):
    sample_x = np.linspace(*xLim, num)
    sample_y = Func(sample_x)
    x_c = np.sum(sample_x * sample_y) / np.sum(sample_y)

    # 用黄金分割法计算质心y坐标
    y_c = np.average(yLim)

    def F_optim(y_try):
        tmp = sample_y.copy()
        idx = np.where(tmp>y_try)
        tmp[idx] = y_try-tmp[idx]
        return np.sum(tmp)

    a = yLim[0]
    b = yLim[1]
    eps = 1e-3
    round = 0
    yt = (a + b) / 2
    test = F_optim(yt)
    # print(a, b, yt, test)
    while np.abs(test) > eps and round<100:
        if test > 0:
            b = yt
        elif test < 0:
            a = yt
        elif test == 0:
            break
        yt = (a + b)/ 2
        test = F_optim(yt)
        # print(a,b,yt,test)
        round +=1
    y_c = yt
    # y_c = np.sum(sample_x * sample_y)/np.sum(sample_x)
    # print(np.sum(sample_x * (sample_y - y_c)))
    return x_c, y_c


if __name__ == '__main__':
    from FuzzyMF import trimf
    from matplotlib import pyplot as plt

    x = np.linspace(0, 20, 400)
    y2 = trimf(x, 3, 8, 15)
    x_c, y_c = center_of_sets(lambda xx: trimf(xx, 3, 8, 15),
                              xLim=(0, 100))
    plt.plot(x, y2, label="Triangular MF")
    plt.scatter(x_c, y_c)
    plt.legend(loc="upper right")
    plt.show()
    print(x_c, y_c)

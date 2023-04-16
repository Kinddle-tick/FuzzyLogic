#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyMF.py
# @Time      :2023/3/9 4:57 PM
# @Author    :Kinddle
import numpy as np
from DelicateFunction import *

Accuracy = 1e-3


def _packet_(*args):
    rtn = []
    for value in args:
        value = np.array(value)
        if value.ndim == 0:
            value = np.array([value])
        rtn.append(value)
    return rtn

def impulsemf(middle=0):
    middle, = _packet_(middle)
    # middle = np.array(middle)
    # if middle.ndim == 0:
    #     middle = np.array([middle])
    # if type(middle) == int:
    #     middle = np.array([middle])

    def wrapper(X):
        X = np.array([X])
        return (X == middle).astype(int)[0]

    eps = np.finfo(np.float32).eps  # 一个很小的数
    D_wrapper = BasicDF(wrapper)
    D_wrapper["max_x"] = middle
    D_wrapper["xLim"] = (middle, middle)
    D_wrapper["yLim"] = (0, 1)
    D_wrapper["Accuracy"] = Accuracy
    return D_wrapper


def gaussmf(Sigma=1, middle=0):
    Sigma,middle = _packet_(Sigma,middle)

    def wrapper(X):
        return np.exp(-(X - middle) ** 2 / (2 * Sigma ** 2))

    D_wrapper = BasicDF(wrapper)
    D_wrapper["max_x"] = middle
    D_wrapper["xLim"] = (middle - 3 * Sigma, middle + 3 * Sigma)
    D_wrapper["yLim"] = (0, 1)
    D_wrapper["Accuracy"] = Accuracy
    return D_wrapper


def sigmf(a=1, center=0):
    a,center = _packet_(a,center)
    if type(a) == int:
        a = np.array([a])
    if type(center) == int:
        center = np.array([center])

    def wrapper(X):
        return 1 / (1 + np.exp(-a * (X - center)))

    D_wrapper = BasicDF(wrapper)
    D_wrapper["yLim"] = (0, 1)
    D_wrapper["Accuracy"] = Accuracy
    return D_wrapper


def trimf(a=0, b=1, c=2):
    # assert a <= b <= c
    return trapmf(a, b, b, c)


def trapmf(a=0, b=1, c=2, d=3):
    assert np.all(np.r_[a - b, b - c, c - d] <= 0)
    a,b,c,d = _packet_(a,b,c,d)

    def wrapper(X):
        X = np.array([X])
        A = np.clip((X - a) / (b - a), 0, 1)
        B = np.clip((d - X) / (d - c), 0, 1)
        np.nan_to_num(A, False, nan=1)
        np.nan_to_num(B, False, nan=1)
        rtn = (A * B)[0]
        return rtn
        # rtn = np.zeros(X.shape)
        #
        # if a == b:
        #     rtn[np.where(X == b)] = 1
        # else:
        #     tmp_idx = np.where(X > a)
        #     rtn[tmp_idx] = ((X - a) / (b - a))[tmp_idx]
        # tmp_idx = np.where(X > b)
        # rtn[tmp_idx] = 1
        # if c == d:
        #     rtn[np.where(X == c)] = 1
        # else:
        #     tmp_idx = np.where(X > c)
        #     rtn[tmp_idx] = (d - X[tmp_idx]) / (d - c)
        # tmp_idx = np.where(X > d)
        # rtn[tmp_idx] = 0
        # return rtn[0]

    D_wrapper = BasicDF(wrapper)
    D_wrapper["max_x"] = (b + c) / 2
    D_wrapper["xLim"] = (a, d)
    D_wrapper["yLim"] = (0, 1)
    D_wrapper["Accuracy"] = Accuracy
    return D_wrapper


def gbellmf(a, b, center):
    a,b,center = _packet_(a,b,center)

    def wrapper(X):
        return 1 / (1 + ((X - center) / a) ** (2 * b))

    D_wrapper = BasicDF(wrapper)
    D_wrapper["max_x"] = center
    # D_wrapper["xLim"] = (middle-3*Sigma, middle+3*Sigma)
    D_wrapper["yLim"] = (0, 1)
    D_wrapper["Accuracy"] = Accuracy
    return D_wrapper


# def maxmin_composition(A, B):
#     C = np.zeros([A.shape[0], B.shape[1]])
#     assert A.shape[1] == B.shape[0]
#     for i in range(A.shape[0]):
#         for j in range(B.shape[1]):
#             tmp_A = A[i, :]
#             tmp_B = B[:, j]
#             C[i, j] = np.max(np.min([tmp_A, tmp_B], axis=0))
#     return C


if __name__ == '__main__':
    tests = [1, np.array([1, 2, 3, 4]) * 0.1, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 0.1]
    funcs = [gaussmf, trimf, trapmf]
    for func in funcs:
        tmp_func = func()
        for x in tests:
            tmp_result = tmp_func(x)
            print(tmp_result, end="\n\n")

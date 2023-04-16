#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyFuzzifier.py
# @Time      :2023/3/27 10:08 AM
# @Author    :Kinddle
import numpy as np


def SingletonFf(X, x_c):
    X = np.array(X)
    rtn = np.zeros(shape=X.shape)
    rtn[X == x_c] = 1
    return rtn


def GaussianFf(X,x_c,Sigma):
    return np.exp(-(X - x_c) ** 2 / (2 * Sigma ** 2))


def TriangularFf(X,x_c,left,right):
    rtn = np.zeros(X.shape)
    assert left <= x_c <= right
    tmp_idx = np.where(X > left)
    rtn[tmp_idx] = (X[tmp_idx] - left) / (x_c - left)
    tmp_idx = np.where(X > x_c)
    rtn[tmp_idx] = (right - X[tmp_idx]) / (right - x_c)
    tmp_idx = np.where(X > right)
    rtn[tmp_idx] = 0
    return rtn
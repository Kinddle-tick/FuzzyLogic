#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzyMF.py
# @Time      :2023/3/9 4:57 PM
# @Author    :Kinddle
import numpy as np


def gaussmf(X, Sigma=1, middle=0):
    return np.exp(-(X - middle) ** 2 / (2 * Sigma ** 2))


def trimf(X, a=0, b=1, c=2):
    X = np.array(X)
    if len(X.shape)==0:
        X= np.array([X])
    rtn = np.zeros(X.shape)
    assert a <= b <= c
    tmp_idx = np.where(X > a)
    rtn[tmp_idx] = (X[tmp_idx] - a) / (b - a)
    tmp_idx = np.where(X > b)
    rtn[tmp_idx] = (c - X[tmp_idx]) / (c - b)
    tmp_idx = np.where(X > c)
    rtn[tmp_idx] = 0
    if a==b:
        rtn[np.where(X==b)]=1
    if c==b:
        rtn[np.where(X==b)]=1
    return rtn


def sigmf(X, a=1, center=0):
    return 1 / (1 + np.exp(-a * (X - center)))


def trapmf(X, a=0, b=1, c=2, d=3):
    X = np.array(X)
    if len(X.shape)==0:
        X= np.array([X])
    rtn = np.zeros(X.shape)
    assert a <= b <= c <= d
    tmp_idx = np.where(X > a)
    rtn[tmp_idx] = (X[tmp_idx] - a) / (b - a)
    tmp_idx = np.where(X > b)
    rtn[tmp_idx] = 1
    tmp_idx = np.where(X > c)
    rtn[tmp_idx] = (d - X[tmp_idx]) / (d - c)
    tmp_idx = np.where(X > d)
    rtn[tmp_idx] = 0
    if a==b:
        rtn[np.where(X==b)]=1
    if c==d:
        rtn[np.where(X==c)]=1
    return rtn


def gbellmf(X, a, b, center):
    return 1 / (1 + ((X - center) / a) ** (2 * b))


def maxmin_composition(A, B):
    C = np.zeros([A.shape[0], B.shape[1]])
    assert A.shape[1] == B.shape[0]
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp_A = A[i,:]
            tmp_B = B[:,j]
            C[i,j] = np.max(np.min([tmp_A,tmp_B],axis=0))
    return C
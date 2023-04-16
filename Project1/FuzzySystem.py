#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzySystem.py
# @Time      :2023/3/28 1:45 PM
# @Author    :Kinddle

import numpy as np
from FuzzyMF import *
from DelicateFunction import BasicDF

class FuzzySystem:
    Flag_Singleton = True  # 是否为Singleton输入 如果是 Fuzzifier 将会被屏蔽
    Flag_Quick = True # 尽可能减少计算量的方法
    Accuracy = 0.1

    def __init__(self, A_dim=3, C_dim=1, A_lim=(0, 10), C_lim=(0, 10), precision=0.1):
        self.Fuzzifier = BasicDF(lambda X: lambda Y: [None])
        self.Defuzzifier = BasicDF(lambda X, Y, Sample: None)
        self.Accuracy = precision
        self.A_dim = A_dim
        self.A_Lim = A_lim
        self.A_MF_List = {}

        self.C_Dim = C_dim
        self.C_Lim = C_lim
        self.C_MF_List = {}

        self.Rule_List = {}  # 类似于(0,1,1)
        self.options = {"composition": "max_prod",  # or "max_min"
                        "multi-": "prod",  # or "min"
                        "implication": "prod",  # or "min"
                        }

    def set_Flag_singleton(self, Flag):
        self.Flag_Singleton = Flag

    def composition(self, *args):
        if self.options["composition"] == "max_prod":
            return np.prod(args)
        if self.options["composition"] == "max_min":
            return np.min(args)

    def mult_(self, *args):
        if self.options["multi-"] == "prod":
            return np.prod(args)
        if self.options["multi-"] == "min":
            return np.min(args)

    def implication(self, *args):
        if self.options["implication"] == "prod":
            return np.prod(args)
        if self.options["implication"] == "min":
            return np.min(args)

    def inference(self, X_):
        X_=np.array(X_)
        if self.Flag_Singleton:
            firing_levels = {}
            Mu_Ax = self.Fuzzifier(X_)
            for A, C in self.Rule_List.items():
                tmp_composition = []
                for i in range(self.A_dim):
                    tmp_composition.append(self.composition(self.A_MF_List[A[i]](X_[i]), Mu_Ax(X_)[i]))
                mult_result = self.mult_(*tmp_composition)
                if mult_result != 0:
                    firing_levels.update({(A, C): mult_result})
            return self.Defuzzifier(firing_levels, self.C_MF_List)

        if not self.Flag_Singleton:
            if not self.Flag_Quick:
                Sample = np.arange(self.A_Lim[0] / self.Accuracy, self.A_Lim[1] / self.Accuracy + 1) * self.Accuracy
                Xs = Sample[:, None].repeat(self.A_dim, axis=1)
                inputs = X_
                firing_levels = {}
                Mu_A = self.Fuzzifier(inputs)(Xs)
                for A, C in self.Rule_List.items():
                    Mu_F = np.vstack([self.A_MF_List[A[i]](Xs[:, i]) for i in range(len(A))]).T
                    tTmp = np.dstack([Mu_A, Mu_F])
                    fl_X = Xs[np.argmax(np.prod(tTmp, axis=2), axis=0), [0, 1, 2]]
                    fl_y = np.prod(np.max(np.prod(tTmp, axis=2), axis=0))
                    # print(f"A:{A},C:{C}:\nFiring Level:({fl_X},{fl_y}) ")
                    if fl_y!=0:
                        firing_levels.update({(A, C): fl_y})
            else:
                # Mu_A = self.Fuzzifier(X_)
                firing_levels = {}
                for A, C in self.Rule_List.items():
                    one_sup_x = np.zeros(len(A))
                    one_sup = np.zeros(len(A))
                    for i in range(len(A)):
                        Mu_F_i = self.A_MF_List[A[i]]
                        Mu_A_i = self.Fuzzifier(X_[i])
                        if self.options["composition"] == "max_prod":
                            New_Mu = Mu_A_i * Mu_F_i
                        elif self.options["composition"] == "max_min":
                            New_Mu = Mu_A_i.min_with(Mu_F_i)
                        else:
                            New_Mu = Mu_A_i * Mu_F_i
                        one_sup_x[i] = New_Mu["max_x"]
                        one_sup[i] = New_Mu(New_Mu["max_x"])
                    fl_X = one_sup_x
                    fl_y = self.mult_(one_sup)
                    if fl_y != 0:
                        firing_levels.update({(A, C): fl_y})

            # firing_levels = {}
            # Mu_Ax = self.Fuzzifier(X_)
            # for A, C in self.Rule_List.items():
            #     tmp_composition = []
            #     for i in range(self.A_dim):
            #         tmp_composition.append(self.composition(self.A_MF_List[A[i]](X_[i]), Mu_Ax(X_)[i]))
            #     mult_result = self.mult_(*tmp_composition)
            #     if mult_result != 0:
            #         firing_levels.update({(A, C): mult_result})
            return self.Defuzzifier(firing_levels, self.C_MF_List)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FuzzySystem.py
# @Time      :2023/3/27 5:06 PM
# @Author    :Kinddle

import numpy as np
from FuzzyMF import *


class FuzzySystem:
    Flag_Singleton = True   # 是否为Singleton输入 如果是 Fuzzifier 将会被屏蔽

    def __init__(self):
        self.Fuzzifier = lambda X: lambda Y:[None]
        self.Defuzzifier = lambda X,Y: None
        self.A_dim = 3
        self.A_MF_List = {}
        self.C_Dim = 1
        self.C_MF_List = {}
        self.Rule_List = {}  # 类似于(0,1,1)
        self.options = {"composition": "max_prod",  # or "max_min"
                        "multi-": "prod",  # or "min"
                        "implication": "prod",  # or "min"
                        }
    def composition(self,*args):
        if self.options["composition"] == "max_prod":
            return np.prod(args)
        if self.options["composition"] == "max_min":
            return np.min(args)
    def mult_(self,*args):
        if self.options["multi-"] == "prod":
            return np.prod(args)
        if self.options["multi-"] == "min":
            return np.min(args)

    def implication(self,*args):
        if self.options["implication"] == "prod":
            return np.prod(args)
        if self.options["implication"] == "min":
            return np.min(args)

    def inference(self, X_):
        if self.Flag_Singleton:
            firing_levels = {}
            Mu_Ax = self.Fuzzifier(X_)
            for A, C in self.Rule_List.items():
                tmp_composition = []
                for i in range(self.A_dim):
                    tmp_composition.append(self.composition(self.A_MF_List[A[i]](X_[i]), Mu_Ax(X_)[i]))
                mult_result = self.mult_(*tmp_composition)
                if mult_result != 0:
                    firing_levels.update({(A,C): mult_result})
            return self.Defuzzifier(firing_levels,self.C_MF_List)

            # return firing_levels

            # return firing_levels



    # def Fuzzifier(self, Input, *args):
    #     return

# class FuzzySystemA(FuzzySystem):
#     def __init__.py(self):
#         super(FuzzySystemA, self).__init__.py()
#
#     def Fuzzifier(self, Input, center,*args):
#         pass

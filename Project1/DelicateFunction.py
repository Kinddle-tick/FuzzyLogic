#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DelicateFunction.py
# @Time      :2023/3/28 1:05 PM
# @Author    :Kinddle
import numpy as np
class BasicDF(dict):
    """
    max_point
    # min_point
    xLim
    yLim
    """

    def __init__(self, function):
        super(BasicDF, self).__init__()
        self.function = function
        # self.Attribute_dict = {}
        self.update({"max_x": None,
                     "xLim": None,
                     "yLim": None,
                     "Accuracy":None})

    # def __call__(self, *args, **kwargs):
    #     return self.function(*args, **kwargs)
    def __call__(self, X):
        return self.function(X)

    def __mul__(self, other):
        # 只做了一维的 累了
        def wrapper(X):
            return self(X)*other(X)
        D_wrapper = BasicDF(wrapper)
        LeftBound = np.max([self["xLim"][0],other["xLim"][0]])
        RightBound = np.min([self["xLim"][1], other["xLim"][1]])
        acc = min(self["Accuracy"], other["Accuracy"])
        if LeftBound > RightBound:  # 全零函数
            D_wrapper.function = lambda x: np.zeros(x.shape if np.array(x).ndim !=0 else 1)
            D_wrapper["xLim"] = (0, 0)
            D_wrapper["max_x"] = 0
            D_wrapper["yLim"] = (0, 0)
            return D_wrapper
        D_wrapper["xLim"] = (LeftBound, RightBound)
        Sample = np.linspace(LeftBound, RightBound, np.ceil((RightBound-LeftBound)/acc).astype(int)+1)
        D_wrapper["max_x"] = Sample[np.argmax(wrapper(Sample))]
        D_wrapper["yLim"] =(0, 1)
        return D_wrapper

    def min_with(self,other):
        def wrapper(X):
            return np.min([self(X), other(X)], axis=0)
        D_wrapper = BasicDF(wrapper)
        LeftBound = np.max([self["xLim"][0],other["xLim"][0]])
        RightBound = np.min([self["xLim"][1], other["xLim"][1]])
        acc = min(self["Accuracy"], other["Accuracy"])
        if LeftBound > RightBound:  # 全零函数
            D_wrapper.function = lambda x: np.zeros(x.shape if np.array(x).ndim !=0 else 1)
            D_wrapper["xLim"] = (0, 0)
            D_wrapper["max_x"] = 0
            D_wrapper["yLim"] = (0, 0)
            return D_wrapper
        D_wrapper["xLim"] = (LeftBound, RightBound)
        Sample = np.linspace(LeftBound, RightBound, np.ceil((RightBound-LeftBound)/acc).astype(int)+1)
        D_wrapper["max_x"] = Sample[np.argmax(wrapper(Sample))]
        D_wrapper["yLim"] = (0, 1)
        return D_wrapper
    def __getitem__(self, item):
        if self.__contains__(item):
            rtn = super(BasicDF, self).__getitem__(item)
            if rtn is None:
                print("该属性未定义")
            else:
                return rtn
        else:
            print("没有这个属性")
            return None


class FunctionDecorator(object):
    def __init__(self, OptName):
        self.optName = OptName

    def __call__(self, func):
        def wrapper(DF, *args, **kwargs):
            # if type(DF) == BasicDF:
            #     if self.optName in DF.keys():
            #         return DF.Attribute_dict[self.optName]
            if self.optName in DF:
                return DF[self.optName]
            else:
                rtn = func(DF, *args, **kwargs)
                DF[self.optName] = rtn
                return rtn
        return wrapper

    # def __str__(self):
    #     print(self, "self.function")

# class logger(object):
#     def __init__.py(self, func):
#         self.func = func
#
#     def __call__(self, *args, **kwargs):
#         print("[INFO]: the function {func}() is running..." \
#               .format(func=self.func.__name__))
#         return self.func(*args, **kwargs)
#
#
# @logger
# def say(something):
#     print("say {}!".format(something))

# class logger(object):
#     def __init__.py(self, level='INFO'):
#         self.level = level
#
#     def __call__(self, func):  # 接受函数
#         def wrapper(*args, **kwargs):
#             print("[{level}]: the function {func}() is running..." \
#                   .format(level=self.level, func=func.__name__))
#             func(*args, **kwargs)
#
#         return wrapper  # 返回函数
#
#
# @logger(level='WARNING')
# def say(something):
#     print("say {}!".format(something))
#
#
# say("hello")
#
# say("hello")

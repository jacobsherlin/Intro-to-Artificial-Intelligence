#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SumofGaussians.py

Created on Tue Sep  3 21:21:15 2019

@author: Scott P. Morton M.S.C.S.
Middle Tn State University

Portions based on C++ code provided by Dr. Joshua Lee Phillips MTSU
Modifed by J.L.P - 2019-09-19 10:18:35
Modifed by J.L.P - 2022-09-29 12:04:22
"""

import numpy as np

class SumofGaussians():
    def __init__(self,dimensions,number_of_centers,rng):
        if (dimensions < 1 or number_of_centers < 1):
            self.centers = None
            return
        self.centers = np.array([rng.uniform(size=dimensions) * 10.0 for i in range(number_of_centers)])
        return
    def Evaluate(self,point):
        return np.sum(np.exp(-np.sum(np.apply_along_axis(lambda x: (point - x)**2.0,1,self.centers),1)))
    def Gradient(self,point):
        return np.sum(-1.0 * np.apply_along_axis(lambda x: np.exp(-np.sum((point-x)**2.0))*(2.0*(point-x)),1,self.centers),0)

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 14:12:58 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler =pd.read_csv("maaslar_yeni.csv")

#Slicing
x= veriler.iloc[:,2:5].values
y= veriler.iloc[:,-1:].values

#Lineer Regresyon
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

import statsmodels.api as sm

model = sm.OLS(lin_reg.predict(x) , x)
#print(model.fit().summary())

from sklearn.metrics import r2_score
print("Lineer Regresyon R^2")
print(r2_score(y, lin_reg.predict(x)))


#Polinomal Regresyon
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

x_poly =poly.fit_transform(x)
print(x_poly)

linerReg= LinearRegression()
linerReg.fit(x_poly, y)

model1 = sm.OLS(linerReg.predict(x_poly), x_poly)
#print(model1.fit().summary())

print("Poly Regresyon R^2")
print(r2_score(y, linerReg.predict(x_poly)))












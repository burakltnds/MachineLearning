# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:53:01 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("maaslar.csv")

#Sliceing
x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

#Lineer
from sklearn.linear_model import  LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x, y)
plt.plot(x, lin_reg.predict(x))
plt.show()
#lineer regresyon saçma durdu bize kırılma noktaları yani polinomal lazım

#Polinomal Regresyon
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

x_poly =poly.fit_transform(x)
print(x_poly)

linerReg= LinearRegression()
linerReg.fit(x_poly, y)
plt.scatter(x, y, color="gray")
plt.plot(x,linerReg.predict(poly.fit_transform(x)),color="blue")
plt.show()

#4.Dereceden Dönüşümü Deneyelim

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)

x_poly =poly.fit_transform(x)
print(x_poly)

linerReg= LinearRegression()
linerReg.fit(x_poly, y)
plt.scatter(x, y, color="red")
plt.plot(x,linerReg.predict(poly.fit_transform(x)),color="yellow")
plt.show()

#tahminler
print("LİNEER MODELİN TAHMİNİ:")
print(lin_reg.predict([[8]]))
print("POLİNOMAL MODELİN TAHMİNİ:")
print(linerReg.predict(poly.fit_transform([[8]])))





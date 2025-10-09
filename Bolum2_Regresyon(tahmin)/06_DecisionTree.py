# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 12:10:59 2025

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

#SVR Veri ölçekleme

from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(y)

from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")

svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli , svr_reg.predict(x_olcekli) ,color="blue")
plt.show()

print("SVR MODELİN TAHMİNİ:")
print(svr_reg.predict([[11]]))

#Decision Tree
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x,y)

plt.scatter(x,y,color="green")
plt.plot(x,dt_reg.predict(x),color="red")
plt.show()

print("KARAR AĞACI MODELİN TAHMİNİ:")
print(linerReg.predict(poly.fit_transform([[9]])))













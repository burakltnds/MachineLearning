# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:50:59 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("satislar.csv")

aylar = veriler[["Aylar"]]

satislar = veriler[["Satislar"]]

from sklearn.model_selection import train_test_split

x_train , x_test , y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0 )

"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
"""
#Model İnşası

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train , y_train)
tahmin = lr.predict(x_test) 

#Gorselleştirelim

x_train = x_train.sort_index()
y_train = y_train.sort_index()



plt.plot(x_train , y_train)
plt.plot(x_test , lr.predict(x_test))
plt.show()
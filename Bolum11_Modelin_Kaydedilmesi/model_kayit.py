# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:10:55 2025

@author: burak
"""

import pandas as pd

veriler = pd.read_csv("satislar.csv")

X = veriler.iloc[:,0:1].values
Y = veriler.iloc[:,1].values

from sklearn import model_selection
X_train , X_test ,Y_train,Y_test = model_selection.train_test_split(X, Y,test_size=0.33)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)

print(lr.predict(X_test))

import pickle 

dosya = "model.kayit"

pickle.dump(lr , open( dosya,"wb") ) 

yuklenen = pickle.load(open(dosya,"rb"))

print(yuklenen.predict(X_test))
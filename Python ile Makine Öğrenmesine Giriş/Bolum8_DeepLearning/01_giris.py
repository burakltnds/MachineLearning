# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 17:15:14 2025

@author: burak
"""

import numpy as np
import pandas as pd

veriler = pd.read_csv("Churn_Modelling.csv")

X = veriler.iloc[:,3:13].values
y = veriler.iloc[:,13].values

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

ohe = ColumnTransformer([("ohe" , OneHotEncoder(dtype=float),[1])] , remainder = "passthrough")

X = ohe.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import keras 
from keras.models import Sequential
from keras.layers import Dense , Input

model = Sequential()

model.add(Input(shape=(11,)))

model.add(Dense(7,activation = "relu"))

model.add(Dense(7,activation = "relu"))

model.add(Dense(1,activation = "sigmoid"))

model.compile(optimizer="adam" , loss= "binary_crossentropy" , metrics=["accuracy"])

model.fit(X_train , y_train , epochs= 50 )

y_pred = model.predict (X_test)

y_pred = (y_pred > 0.5) #true false olarak döndürcez

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)












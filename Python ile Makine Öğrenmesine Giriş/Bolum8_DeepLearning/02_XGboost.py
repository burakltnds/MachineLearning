# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:43:33 2025

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

from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train , y_train )

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)












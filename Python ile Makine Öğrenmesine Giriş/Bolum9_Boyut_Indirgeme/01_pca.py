# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:28:56 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("wine.csv")

X = veriler.iloc[:,0:13]
y = veriler.iloc[:,13]

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components= 2) #kaç boyuta indirgeyeceğin

X_train2 = pca.fit_transform(X_train)

X_test2 = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

model2 = LogisticRegression(random_state=0)

model2.fit(X_train2, y_train)

#Tahminler

y_pred = model.predict(X_test)

y_pred2 = model2.predict(X_test2)


from sklearn.metrics import confusion_matrix

print("Pca'sız")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Pca'lı")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
print("Karşılaştırma")
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)






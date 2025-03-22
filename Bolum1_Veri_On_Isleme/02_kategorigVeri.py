# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 18:04:50 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")

print(veriler)

#eksik veriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan , strategy="mean")

yas = veriler.iloc[:,1:4].values

print(yas)

imputer = imputer.fit(yas[:,1:4]) #öğren
yas[:,1:4] = imputer.transform(yas[:,1:4]) #uygula(ortalamayı boş yerlere ekler)
print(yas) 

#Kategorik

ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #sadece 2 veri varsa faydalı
print(ulke)

ohe = preprocessing.OneHotEncoder() #çoklu verilerde faydalı

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)


























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

#birleşim

sonuc = pd.DataFrame(data=ulke , index=range(22) ,columns=["fr","tr","us"] )

sonuc2 = pd.DataFrame(data=yas , index=range(22) ,columns=["boy","kilo","yas"] )

cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=cinsiyet , index=range(22) ,columns=["cinsiyet"] )


s= pd.concat([sonuc,sonuc2] , axis=1)
s2= pd.concat([s,sonuc3] , axis=1)
print(s2)

#verileri ayırma
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s, sonuc3 ,test_size=0.33,random_state=0) 

#öznitelik ölçekleme

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train =sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)












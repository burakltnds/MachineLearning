# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:21:40 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("odev_tenis.csv")

hava = veriler.iloc[:,0:1].values
sonuc = veriler.iloc[:,-1:].values
ruzgar = veriler.iloc[:,-2:-1].values.astype(int)
diger = veriler.iloc[:,1:3].values

#Encode

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
sonuc[:,0] = le.fit_transform(sonuc[:,0])


ohe =preprocessing.OneHotEncoder()
hava =ohe.fit_transform(hava).toarray()

#Dataframeleri oluşturma

df1 = pd.DataFrame(data=hava ,index=range(14)  ,columns=["Overcast","Rainy","Sunny"] )
df2 = pd.DataFrame(data=diger,index=range(14) ,columns=["Temprature" ,"Humidity" ] )
df3 = pd.DataFrame(data=sonuc,index=range(14) ,columns=["Sonuç(0=hayır)"] )

#Birleştirme

s=pd.concat([df1,df2] , axis=1)
s2=pd.concat([s,df3] , axis=1)

#Veri Bölme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s, df3 ,test_size=0.33,random_state=0)

#Çoklu Değişken

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predict = regressor.predict(x_test)

#kontrol aşaması

import statsmodels.api as sm

x= np.append(arr=np.ones((14,1)).astype(int), values=s2.iloc[:,:-1] ,axis=1)

x_list = s2.iloc[:,[0,1,2,3,4]].values
x_list = np.array(x_list,dtype=float)
model = sm.OLS(s2.iloc[:,-1:].values.astype(float),x_list).fit()
print(model.summary())















# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:20:46 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")

Yas = veriler.iloc[:,1:4].values


# Encode ülke
ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
ulke =ohe.fit_transform(ulke).toarray()

# Encode cinsiyet
c = veriler.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()



#Dizilerin DF ye dönüşümü 

sonuc = pd.DataFrame(data=ulke , index=range(22) , columns=["FR","TR","US"])

sonuc3 = pd.DataFrame(data=c[:,:1] , index=range(22), columns=["Cinsiyet"])

sonuc2 = pd.DataFrame(data = Yas , index=range(22) ,columns=["Boy" , "Kilo" ,"Yaş"])

#Birleştirme

s=pd.concat([sonuc,sonuc2],axis=1)

s2 = pd.concat([s,sonuc3] , axis=1)


#Veri Bölme
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s, sonuc3 ,test_size=0.33,random_state=0)

#Çoklu Değişken Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predict = regressor.predict(x_test)
 

 
#Boy Tahmini Yapalım

boy = s2.iloc[:,3:4].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)
x_train,x_test,y_train,y_test = train_test_split(veri, boy ,test_size=0.33,random_state=0)

regressor1 = LinearRegression()
regressor1.fit(x_train, y_train)

y_predict1 = regressor1.predict(x_test)

import statsmodels.api as sm

x=np.append(arr=np.ones((22,1)).astype(int) , values=veri ,axis=1)
#1 lerden oluşan bir dizi ekler a+b(yas) +c(boy) + e deki e yi temsil eder

x_list = veri.iloc[:,[0,1,2,3,4,5]].values
model =sm.OLS(boy,x_list).fit()
print(model.summary())

#4.en yüksek çıktı eliyoruz

x_list = veri.iloc[:,[0,1,2,3,5]].values
model =sm.OLS(boy,x_list).fit()
print(model.summary())









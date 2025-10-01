# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 15:39:08 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri çekme

veriler = pd.read_csv("Iris.csv")

x = veriler.iloc[:,0:4]
y = veriler.iloc[:,4:]

# Veri bölme

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split(x , y, test_size=0.33 ,random_state=0)

"""
# Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test =sc.fit_transform(x_test)
"""

## Lojistik regresyon

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(x_train , y_train)

y_predict = logr.predict(x_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)

print('Lojistik Regresyon')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict))
print(cm)

## KNN EUCLADIAN 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1 , metric='euclidean')

knn.fit(x_train, y_train)

y_predict2 = knn.predict(x_test)

cm2 = confusion_matrix(y_test, y_predict2) 

print('KNN Euclidean')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict2))
print(cm2)

## KNN Manhattan 

knn2 = KNeighborsClassifier(n_neighbors=1 , metric='manhattan')

knn2.fit(x_train, y_train)

y_predict3 = knn2.predict(x_test)

cm3 = confusion_matrix(y_test, y_predict3) 

print('KNN Manhattan')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict3))
print(cm3)

# SVM
from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(x_train ,y_train )
y_predict4 =svc.predict(x_test)

cm4 =confusion_matrix(y_test, y_predict4)

print('SVM')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict4))
print(cm4)

# Bernoulli Naive Byes (ikili veriler için)

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_predict5 = bnb.predict(x_test)

cm5 = confusion_matrix(y_test, y_predict5) 

print('Bernoulli Naive Byes')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict5))
print (cm5)

# Gaussian Naive Byes (sürekli veriler için)

from sklearn.naive_bayes import GaussianNB

gnb =GaussianNB()

gnb.fit(x_train, y_train)

y_predict6 = gnb.predict(x_test)

cm6 = confusion_matrix(y_test, y_predict6)

print('Gaussian Naive Byes')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict6))
print(cm6)


# Multinominal Naive Byes (sayma tabanlı veriler) 
# Negatif verilerle çalışamaz

"""
from sklearn.naive_bayes import MultinomialNB

mnb =MultinomialNB()

mnb.fit(x_train, y_train)

y_predict7 = mnb.predict(x_test)

cm7 = confusion_matrix(y_test, y_predict7)

print('MN Naive Byes')
print(cm7)

"""

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_predict8 = dt.predict(x_test)

cm8 = confusion_matrix(y_test, y_predict8)

print('Decision Tree')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict8))
print(cm8)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_predict9 = rf.predict(x_test)

cm9 = confusion_matrix(y_test , y_predict9)

print('Random Forest')
print("Doğruluk Oranı:", accuracy_score(y_test, y_predict9))
print(cm9)



















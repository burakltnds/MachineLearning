# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:53:20 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib as plt

veriler = pd.read_csv("veriler.csv")

x=veriler.iloc[:,1:4].values

y=veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test =train_test_split(x, y,test_size=0.33 , random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10 ,criterion='entropy')

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print ('RF')

print(cm)



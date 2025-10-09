# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:18:30 2025

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



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components= 2 )



X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)

model.fit(X_train_lda, y_train)

y_pred = model.predict(X_test_lda)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



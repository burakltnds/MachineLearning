# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 12:40:58 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

x = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 4 , init = "k-means++")

kmeans.fit(x)

print(kmeans.cluster_centers_)

sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i , init="k-means++" , random_state=123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)

# wcss değerleri
plt.plot(range(1,11) , sonuclar)
plt.show()


y_predict = kmeans.fit_predict(x)
print(y_predict)
plt.scatter(x[y_predict == 0,0], x[y_predict == 0,1] ,s=100 , c="red")
plt.scatter(x[y_predict == 1,0], x[y_predict == 1,1] ,s=100 , c="blue")
plt.scatter(x[y_predict == 2,0], x[y_predict == 2,1] ,s=100 , c="green")
plt.scatter(x[y_predict == 3,0], x[y_predict == 3,1] ,s=100 , c="yellow")
plt.show()


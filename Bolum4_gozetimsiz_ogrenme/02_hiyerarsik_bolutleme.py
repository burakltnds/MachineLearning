# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:39:22 2025

@author: burak
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

x = veriler.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering

ag = AgglomerativeClustering(n_clusters=4 , metric = "euclidean" , linkage="ward")

y_predict = ag.fit_predict(x)

print(y_predict)

plt.scatter(x[y_predict == 0,0], x[y_predict == 0,1] ,s=100 , c="red")
plt.scatter(x[y_predict == 1,0], x[y_predict == 1,1] ,s=100 , c="blue")
plt.scatter(x[y_predict == 2,0], x[y_predict == 2,1] ,s=100 , c="green")
plt.scatter(x[y_predict == 3,0], x[y_predict == 3,1] ,s=100 , c="yellow")
plt.show()

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(x , method="ward"))
plt.grid(axis="y")
plt.show()
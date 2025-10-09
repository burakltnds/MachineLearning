# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 17:52:37 2025

@author: burak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("spotify.csv")

features = ["popularity","danceability","energy","tempo"]

x =veriler[features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 
x_scaled =scaler.fit_transform(x)

from sklearn.cluster import KMeans , AgglomerativeClustering
import scipy.cluster.hierarchy as sch

"""
clust = []

for i in range (1,10):
    kmeans = KMeans(n_clusters=i, init = "k-means++" , random_state=10)
    kmeans.fit(x)
    clust.append(kmeans.inertia_)

plt.plot(range(1,10) , clust)
plt.grid(axis= "y" )
plt.show()
"""

kmeans = KMeans(n_clusters=4 , init="k-means++")
kmeans.fit(x)

y_predict = kmeans.fit_predict(x)

plt.scatter(x_scaled[y_predict == 0,0], x_scaled[y_predict == 0,1] ,s=100 , c="red" ,label = "1")
plt.scatter(x_scaled[y_predict == 1,0], x_scaled[y_predict == 1,1] ,s=100 , c="blue",label = "2")
plt.scatter(x_scaled[y_predict == 2,0], x_scaled[y_predict == 2,1] ,s=100 , c="green",label = "3")
plt.scatter(x_scaled[y_predict == 3,0], x_scaled[y_predict == 3,1] ,s=100 , c="gray",label = "4")
plt.show()



from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[y_predict == 0, 0], x_pca[y_predict == 0, 1], s=50, c="red", label="Cluster 0")
plt.scatter(x_pca[y_predict == 1, 0], x_pca[y_predict == 1, 1], s=50, c="blue", label="Cluster 1")
plt.scatter(x_pca[y_predict == 2, 0], x_pca[y_predict == 2, 1], s=50, c="green", label="Cluster 2")
plt.scatter(x_pca[y_predict == 3, 0], x_pca[y_predict == 3, 1], s=50, c="gray", label="Cluster 3")
plt.legend()
plt.title("KMeans Clusters (PCA 2D)")
plt.show()

# hiyerarşik (burada çalışmayacak büyük verilerde iyi çalışmaz)

ag = AgglomerativeClustering(n_clusters=4 , metric="euclidean" , linkage="ward")

y_predict1 = ag.fit_predict(x)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[y_predict1 == 0, 0], x_pca[y_predict1 == 0, 1], s=50, c="red", label="Cluster 0")
plt.scatter(x_pca[y_predict1 == 1, 0], x_pca[y_predict1 == 1, 1], s=50, c="blue", label="Cluster 1")
plt.scatter(x_pca[y_predict1 == 2, 0], x_pca[y_predict1 == 2, 1], s=50, c="green", label="Cluster 2")
plt.scatter(x_pca[y_predict1 == 3, 0], x_pca[y_predict1 == 3, 1], s=50, c="gray", label="Cluster 3")
plt.legend()
plt.title("HR Clusters (PCA 2D)")
plt.show()

##veriyi küçültmek gerekir yoksa çalışmayacaktır






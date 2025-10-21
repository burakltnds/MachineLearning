# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:01:03 2025

@author: burak
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import random

veriler = pd.read_csv("ads CTR optimisation.csv")

N = 10000 #tıklama sayısı
d = 10 # ilan sayısı
toplam = 0 #toplam odul sayısı
secilenler = []
birler = [0] * d
sifirlar = [0] * d

for n in range(0,N):
    ad = 0 # seçilen ilan
    max_th = 0
    for i in range(0 , d ): # her ilanı incele en yüksek ucb değerini bul
        randbeta = random.betavariate(birler [i] + 1 , sifirlar[i] +1 )
        if randbeta > max_th:
            max_th=randbeta
            ad=i        
    secilenler.append(ad)
    odul = veriler.values[n , ad]
    if odul ==1 :
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad] = sifirlar[ad] + 1    
    toplam = toplam + odul
    
print("toplam ödül")
print(toplam)

plt.hist(secilenler)
plt.show()
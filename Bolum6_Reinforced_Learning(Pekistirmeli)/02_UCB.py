# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:36:51 2025

@author: burak
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math

veriler = pd.read_csv("ads CTR optimisation.csv")

N = 10000 #tıklama sayısı
d = 10 # ilan sayısı
oduller = [0] * d #butun ilanların odulu ilk başta 0 , Ri(n)
toplam = 0 #toplam odul sayısı
tiklamalar = [0] * d #bir ana kadar olan tıklama sayısı , Ni(n)
secilenler = []

for n in range(0,N):
    ad = 0 # seçilen ilan
    max_ucb = 0
    for i in range(0 , d ): # her ilanı incele en yüksek ucb değerini bul
        if(tiklamalar[ i ] > 0 ):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt( 3 / 2 * math.log(n) / tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N * 10
        
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i

    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar [ad] + 1
    odul = veriler.values[n , ad]
    oduller [ad] = oduller[ad] + odul
    toplam = toplam + odul
    
print("toplam ödül")
print(toplam)

plt.hist(secilenler)
plt.show()


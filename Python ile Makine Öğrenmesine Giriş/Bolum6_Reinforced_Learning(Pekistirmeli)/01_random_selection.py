# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:24:15 2025

@author: burak
"""

import numpy as n
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("ads CTR optimisation.csv")

import random

N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # n. satırdaki = 1 ise ödül
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()    
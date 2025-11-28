# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 15:31:17 2025

@author: burak
"""

#%% IMPORT LIB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Input
from tensorflow.keras.utils import to_categorical

#%% DATA CREATE
def assign__treatment_1 (blood_pressure , blood_sugar , sympton):
    return (blood_pressure < 120 ) & (blood_sugar < 100) & (sympton <= 1)
def assign__treatment_2 (blood_pressure , blood_sugar , sympton):
    return (blood_pressure > 120 ) & (blood_sugar < 140) & (sympton == 1)
def assign__treatment_3 (blood_pressure , blood_sugar , sympton):
    return (blood_pressure > 140 ) & (blood_sugar >= 150) & (sympton == 2)

num_samples = 1000
age = np.random.randint(20,80, size = num_samples)
gender = np.random.randint(0,2,size = num_samples)
disease = np.random.randint(0 , 4 , size = num_samples)
sympton_fever = np.random.randint(0 , 3 , size = num_samples)
sympton_cough = np.random.randint(0 , 3 , size = num_samples)
sympton_headache = np.random.randint(0 , 3 , size = num_samples)
blood_pressure = np.random.randint(90 , 180 , size = num_samples)
blood_sugar = np.random.randint(70 , 200 , size = num_samples)
previous_treatment_responce = np.random.randint(0 , 3 , size = num_samples)

sympton = sympton_cough + sympton_fever + sympton_headache

treatment_plan = np.zeros(num_samples)

for i in range(num_samples):
    if assign__treatment_1(blood_pressure[i], blood_sugar[i], sympton[i]):
        treatment_plan[i] = 0
    elif assign__treatment_2(blood_pressure[i], blood_sugar[i], sympton[i]):
        treatment_plan[i] = 1
    elif assign__treatment_3(blood_pressure[i], blood_sugar[i], sympton[i]):
        treatment_plan[i] = 2

data = pd.DataFrame({
    "age" : age ,
    "gender" : gender ,
    "disease" : disease ,
    "sympton_fever" : sympton_fever ,
    "sympton_cough"  : sympton_cough ,
    "sympton_headache" : sympton_headache ,
    "blood_pressure" : blood_pressure ,
    "blood_sugar" : blood_sugar ,
    "previous_treatment_responce" : previous_treatment_responce,
    "sympton" : sympton ,
    "treatment_plan" : treatment_plan })

#%% TRAIN WITH ANN
X = data.drop(["treatment_plan"] , axis=1).values
y = to_categorical(data["treatment_plan"] , num_classes=3)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3 ,random_state=42)

model = Sequential()
model.add(Input(shape = (X_train.shape[1],)))
model.add(Dense(32,activation = "relu"))
model.add(Dense(64,activation = "relu"))
model.add(Dense(3,activation = "softmax"))

model.compile(optimizer = "adam" , loss="categorical_crossentropy" , metrics = ["accuracy"])

history = model.fit(X_train , y_train , epochs = 20 , validation_data = (X_test , y_test), batch_size = 32)

#%% EVALUATION
val_loss , val_accuracy = model.evaluate(X_test , y_test)
print(f"val_accuracy: {val_accuracy} , val_loss: {val_loss}")

h = history.history
plt.subplot(1,2,1)
plt.plot(h["accuracy"] , "bo-" , label = "Train Acc")
plt.plot(h["val_accuracy"] , "r^-" , label = "Val Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(h["loss"] , "bo-" , label = "Train Loss")
plt.plot(h["val_loss"] , "r^-" , label = "Val Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
































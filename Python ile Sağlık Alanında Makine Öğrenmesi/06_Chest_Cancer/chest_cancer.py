# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:48:23 2025

@author: burak
"""

#%% IMPORT LIB 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D,Input , MaxPool2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import os

from tqdm import tqdm

#%% LOAD DATA
labels = ["PNEUMONIA" , "NORMAL"]
img_size = 150 #150x150

def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir , label)
        class_num = labels.index(label) #pneumonia 0 normal 1
        for img in tqdm([f for f in os.listdir(path) if f.endswith('.jpeg') or f.endswith('.jpg')]):
            print(img)
            try:
                img_arr = cv2.imread(os.path.join(path, img) , cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    print("Read İmage Error")
                    continue
                resized_arr = cv2.resize(img_arr , (img_size , img_size))
                
                data.append([resized_arr , class_num])
            except Exception as e:
                print("Hata:" , e)
    return np.array(data , dtype = object)    

train = get_training_data("chest_xray/train")
test  = get_training_data("chest_xray/test")
val   = get_training_data("chest_xray/val")

# %% DATA VISUALIZATION AND PREPROCESSING
l = []
for i in train:
    if(i[1] == 0):
        l.append("PNEUMONIA")
    else:
        l.append("NORMAL")

sns.countplot(x=l)
plt.show()

x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

for feature , label in train:
    x_train.append(feature)
    y_train.append(label)

for feature , label in test:
    x_test.append(feature)
    y_test.append(label)

for feature , label in val:
    x_val.append(feature)
    y_val.append(label)

plt.figure()
plt.imshow(train[0][0] , cmap="gray")
plt.title(labels[train[0][1]])
plt.show()

# %% Normalizasyon

#değerler ztn 0 ile 255 arasında 255 e bölersek 0 ile 1 arasında olur

x_train = np.array(x_train) / 255
x_test  = np.array(x_test) / 255
x_val   = np.array(x_val) / 255


#reshape (-1 oto yapmasını sağlar sonuna 1 eklemezsek dl ye giremez)
x_train = x_train.reshape(-1 , img_size , img_size ,1)
x_test  = x_test.reshape(-1 , img_size , img_size ,1) 
x_val   = x_val.reshape(-1 , img_size , img_size ,1)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


# %% DATA AUGMENTATION

datagen = ImageDataGenerator(
    featurewise_center = False  ,                # veri setinde genel ortalama 0
    samplewise_center  = False  ,                # her ormeğin ortalamsı 0 
    featurewise_std_normalization = False,       #veriyi verinin standart sapmasına böler
    samplewise_std_normalization = False ,       #her örneği kendi standart sapmasına böler
    zca_whitening = False                ,       #zca beyazlatma yöntemi , korelasyonu azaltır
    rotation_range = False               ,       #resimleri x derece rastgele döndürür
    zoom_range = 0.2                     ,       #rastgele yakınlaştırma işlemi
    width_shift_range = 0.1              ,       #resimleri yatay olarak rastgele kaydırır
    height_shift_range = 0.1             ,       #resimleri dikey olarak rastgele kaydırır
    horizontal_flip = True               ,       #resimleri rastgele yatay çevirir 
    vertical_flip = True                         #resimleri rastgele dikey çevirir
    )
datagen.fit(x_train)

# %% CREATE DEEP LEARNING MODEL AND TRAIN
"""
Feature Extraction Blok:
    con2d - Normalizasyon - MaxPooling
    con2d - dropout - Normalizasyon - MaxPooling
    con2d - Normalizasyon - MaxPooling
    con2d - dropout - Normalizasyon - MaxPooling
    con2d - dropout - Normalizasyon - MaxPooling 
Classification Blok:
    flatten - dense - dropout - dense(output)
Compiler:
    optimizer(adam) , loss(binary cross entr.) , metric(accuracy)
"""

model = Sequential()
model.add(Input(shape=(150, 150, 1)))
model.add(Conv2D(32, (7,7), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2) , strides = 2 , padding= "same"))

model.add(Conv2D(64, (5,5), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2) , strides = 2 , padding= "same"))

model.add(Conv2D(64, (5,5), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2) , strides = 2 , padding= "same"))

model.add(Flatten())
model.add(Dense(units=128 , activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = "sigmoid"))

model.compile(optimizer = "rmsprop" ,loss = "binary_crossentropy",metrics = ["accuracy"] )
model.summary()

learning_rate_reducation = ReduceLROnPlateau(monitor = "val_accuracy" , patience = 2 , verbose = 1 , factor = 0.3 , min_lr = 0.000001)
epoch_number = 3
history = model.fit(datagen.flow(x_train,y_train ,batch_size = 32) , epochs = epoch_number , validation_data = datagen.flow(x_test , y_test) , callbacks = [learning_rate_reducation] )
print("Loss of Model:" , model.evaluate(x_test , y_test)[0])
print("Accuracy of Model:" , model.evaluate(x_test , y_test)[1]*100)

# %% EVALUATION
epochs = [i for i in range(epoch_number)]


fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 

train_acc = history.history["accuracy"]
train_loss = history.history["loss"]

val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]

ax[0].plot(epochs, train_acc, "go-", label="Training Accuracy")
ax[0].plot(epochs, val_acc, "ro-", label="Validation Accuracy") 

ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Training and Validation Accuracy") 

ax[1].plot(epochs, train_loss, "go-", label="Training Loss")
ax[1].plot(epochs, val_loss, "ro-", label="Validation Loss")

ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].set_title("Training and Validation Loss")


plt.tight_layout()
plt.show()

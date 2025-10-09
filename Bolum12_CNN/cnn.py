# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 17:19:07 2025

@author: burak
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

#1 Convolution evrim
model.add (Convolution2D( 32, 3, 3  , input_shape = (64,64,3),activation ="relu"))

#2 Havuzlama
model.add(MaxPooling2D(pool_size=(2,2)))

#3 Bidaha evrim + havuzlama(istediğin kadar yapabilirsin)
model.add (Convolution2D( 32, 3, 3,activation ="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#4 düzleme
model.add(Flatten())

#5 Neural Network
model.add(Dense( units = 128 , activation = "relu"))
model.add(Dense( units = 128 , activation = "relu"))
model.add(Dense( units = 128 , activation = "relu"))
model.add(Dense( units = 128 , activation = "relu"))
model.add(Dense( units = 1 , activation = "sigmoid"))

model.compile(optimizer="adam" , loss="binary_crossentropy" , metrics= ["accuracy"])

#6 CNN

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                 rescale = 1./255,
                 shear_range = 0.2,
                 zoom_range = 0.2 ,
                 horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1./255,
                 shear_range = 0.2,
                 zoom_range = 0.2 ,
                 horizontal_flip = True) 

train_set = train_datagen.flow_from_directory("veriler/training_set",
                                              target_size = (64,64) ,
                                              batch_size = 64 ,
                                              class_mode = "binary"
                                              )

test_set = test_datagen.flow_from_directory("veriler/test_set",
                                              target_size = (64,64) ,
                                              batch_size = 64 ,
                                              class_mode = "binary"
                                              )




model.fit(
    train_set ,
    steps_per_epoch = 10000,
    epochs = 20,
    validation_data = test_set,
    validation_steps = 5000
    )


import numpy  as np
import pandas as pd

test_set.reset ()
pred = model.predict(test_set , verbose = 1)

pred [pred > 0.5] = 1 
pred [pred <= 0.5] = 0


test_labels = []

for i in range(len(test_set)):
    test_labels.extend(np.array(test_set[i][1]))
    
print('test_labels')
print(test_labels)


dosyaisimleri = test_set.filenames

sonuc = pd.DataFrame()
sonuc['dosyaisimleri']= dosyaisimleri
sonuc['tahminler'] = pred
sonuc['test'] = test_labels   

from sklearn.metrics import confusion_matrix


cm = confusion_matrix(test_labels, pred)
print (cm)





print(train_set.class_indices)

from keras.preprocessing import image

# Dosya adları
image1 = "barbara.jpeg"
image2 = "kenan.jpeg"

# Görselleri yükle
image11 = image.load_img(image1, target_size=(64, 64))
image22 = image.load_img(image2, target_size=(64, 64))

# Görselleri diziye çevir
x = image.img_to_array(image11)
y = image.img_to_array(image22)

# Batch boyutu ekle
x = np.expand_dims(x, axis=0)
y = np.expand_dims(y, axis=0)

# Tahmin yap
result1 = model.predict(x)
result2 = model.predict(y)

# Tahminleri yorumla
print("Barbara:", "Kadın" if result1[0][0] == 1 else "Erkek")
print("Kenan:", "Kadın" if result2[0][0] == 1 else "Erkek")








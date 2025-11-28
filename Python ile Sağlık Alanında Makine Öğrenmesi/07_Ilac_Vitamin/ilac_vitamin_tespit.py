# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:46:05 2025

@author: burak
"""

# %% IMPORT LIB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense , Dropout  , Rescaling
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from pathlib import Path
import os.path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# %% LOAD DATASET
dataset = "Drug Vision/Data Combined"
image_dir = Path(dataset)

filepaths = list(image_dir.glob(r"**/*.jpg")) 

labels = list(map(lambda x: Path(x).parent.name , filepaths))

filepaths = pd.Series(filepaths , name = "filepath").astype(str)

labels = pd.Series(labels , name="label") 

image_df = pd.concat([filepaths , labels] , axis=1)

# %% DATA VISUALIZATION
random_index = np.random.randint(0 , len(image_df) , 16)
fig , axes = plt.subplots(nrows = 4, ncols = 4 , figsize=(8,8))

for i , ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    ax.set_title(image_df.label[random_index[i]])
    
plt.tight_layout()
plt.show()

# %% DATA PREPROCESSING
# Train test split
train_df , test_df = train_test_split(image_df, test_size=0.25 , shuffle=True , random_state=42)

# Data Augmentation
train_generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input ,
    validation_split = 0.25 
    )
test_generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input ,
    )

train_images = train_generator.flow_from_dataframe(
    dataframe = train_df , 
    x_col = "filepath",
    y_col = "label",
    target_size = (224,224),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 64 ,
    shuffle = True ,
    seed = 42 ,
    subset= "training"
    )

val_images = train_generator.flow_from_dataframe(
    dataframe = train_df , 
    x_col = "filepath",
    y_col = "label",
    target_size = (224,224),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 64 ,
    shuffle = True ,
    seed = 42 ,
    subset= "validation"
    )

test_images = test_generator.flow_from_dataframe(
    dataframe = test_df , 
    x_col = "filepath",
    y_col = "label",
    target_size = (224,224),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 64 ,
    shuffle = False ,
    )

# resize , rescale
resize_and_rescale = tf.keras.Sequential(
    [layers.Resizing(224,224) ,
     layers.Rescaling(1./255)])

# %% TRANSFER LEARNING
pretrained_model = tf.keras.applications.MobileNetV2(  # önceden eğitilmiş model
    input_shape = (224 , 224 , 3) , # girdilerin boyutu
    include_top = False ,           # mobile net sınıflandırma katmanını dahil etmez
    weights = "imagenet" ,          # hangi veri setiyle eğitim
    pooling = "avg"
    )

pretrained_model.trainable = False

# Checkpoints

checkpoint_path = "best_model.weights.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only = True ,
    monitor = "val_accuracy" ,
    save_best_only = True
    )

early_stopping = EarlyStopping(
    monitor = "val_loss" ,
    patience = 5 ,
    restore_best_weights = True
    )

# Training Model
inputs = pretrained_model.input
x = resize_and_rescale(inputs)
x = Dense(256 , activation="relu")(pretrained_model.output)
x = Dropout(0.2) (x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(10 , activation="softmax")(x)
model = Model(inputs = inputs , outputs = outputs)

model.compile(optimizer = Adam(0.0001) , loss = "categorical_crossentropy" , metrics = ["accuracy"])
 
history = model.fit(
     train_images ,
     steps_per_epoch = len(train_images) ,
     validation_data = val_images ,
     validation_steps = len (val_images),
     epochs = 8 ,
     callbacks = [early_stopping , checkpoint_callback]
     )

# %% MODEL EVALUATE

results = model.evaluate(test_images , verbose = 1)

print("Test Loss:" , results[0] )
print("Test Accuracy:" , results[1] )

epochs = range (1 , len(history.history["accuracy"]) + 1 )

hist = history.history
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs , hist["accuracy"] , "bo-" , label = "Training Accuracy")
plt.plot(epochs , hist["val_accuracy"] , "r^-" , label = "Valudation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs , hist["loss"] , "bo-" , label = "Training Loss")
plt.plot(epochs , hist["val_loss"] , "r^-" , label = "Valudation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

#
pred = model.predict(test_images)
pred = np.argmax(pred , axis=1)

labels = (train_images.class_indices)
labels = dict ((v,k) for k,v in labels.items())
pred   = [labels[k] for k in pred] 

random_index = np.random.randint(0 , len(test_df) ,15)
fig , axes = plt.subplots(nrows = 5, ncols = 3 , figsize=(11,11))

for i , ax in enumerate(axes.flat):
    img_path = test_df.filepath.iloc[random_index[i]]
    true_label = test_df.label.iloc[random_index[i]]
    prediction_label = pred[random_index[i]]
    ax.imshow(plt.imread(img_path))
    if true_label == prediction_label:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {true_label}\nPred: {prediction_label}", color=color)
    
plt.tight_layout()
plt.show()

y_test = list(test_df.label)
print (classification_report(y_test,pred))
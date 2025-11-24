# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 19:01:05 2025

@author: burak
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor #kayıp verileri giderecek

from sklearn.metrics import confusion_matrix , classification_report , accuracy_score

import warnings
warnings.filterwarnings("ignore")

#load and EDA
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

df = df.drop(["id"] , axis = 1)

df.info()

describe = df.describe()

#stroke etkisinin dağılımı
plt.figure()
sns.countplot(x="stroke" , data=df)
plt.title("Distribution of Stroke")
plt.show()

"""
4800 -> 0
250 -> 1

dengesiz veri seti 

 - stroke (1) sayısını arttırmak için veri toplayabiliriz
 - down sampling (0) sayısını azaltmak 

"""

#Missing Value --> DT ile missing valuelar doldurulacak

df.isnull().sum()

#yaş ve cinsiyet kullanarak bmi doldurulabilir

DT_bmi_pipe = Pipeline(steps=[
    ("scale" , StandardScaler()),
    ("dtr" , DecisionTreeRegressor())
    ] )

X = df[["gender" , "age" , "bmi"]].copy()

X.gender = X.gender.replace({"Male" : 0 , "Female" : 1 , "Other" : -1 }).astype(np.uint8)

missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
y = X.pop("bmi")

DT_bmi_pipe.fit(X,y)

predict_bmi = pd.Series(DT_bmi_pipe.predict(missing[["gender" , "age"]]), index=missing.index)

df.loc[missing.index , "bmi"] = predict_bmi 


#Model prediction: encoding , train test split
df["gender"] = df["gender"].replace({"Male" : 0 , "Female" : 1 , "Other" : -1 }).astype(np.uint8)
df["Residence_type"] = df["Residence_type"].replace({"Rural" : 0 , "Urban" : 1}).astype(np.uint8)
df["work_type"] = df["work_type"].replace({"Private" : 0 , "Self-employed" : 1 , "Govt_job" : 2,"children":-1,"Never_worked":-2 }).astype(np.uint8)

X = df[["gender" , "age","hypertension", "heart_disease","work_type","Residence_type","avg_glucose_level","bmi"]]
y = df[["stroke"]]

X_train ,X_test, y_train,y_test = train_test_split(X,y ,test_size=0.33 , random_state=42)

logreg_pipe = Pipeline(steps= [
    ("scale" ,StandardScaler()) ,
    ("LR" , LogisticRegression())
    ])

logreg_pipe.fit(X_train, y_train)

y_pred = logreg_pipe.predict(X_test)

print("Accuracy" , accuracy_score(y_test, y_pred))

print("CM \n" , confusion_matrix(y_test, y_pred))

print("Classi Report:" , classification_report(y_test, y_pred))

"""
CM 
 [[1590    1]
 [  96    0]]
Class Imbalance Problem
"""

#model save and load

import joblib

"""
joblib.dump(logreg_pipe , "log_reg_model.pkl")
"""

loaded_log_reg_pipe = joblib.load("log_reg_model.pkl")

new_patient_data = pd.DataFrame(
    {
     "gender" : [1] ,
     "age" : [46] ,
     "hypertension" : [1] ,
     "heart_disease" : [0] ,
     "work_type" : [0] ,
     "Residence_type" : [0] ,
     "avg_glucose_level" : [70] ,
     "bmi" : [25] ,
     }
    )

new_patient_data_result = loaded_log_reg_pipe.predict(new_patient_data)

#olasılıksal tahmin

new_patient_data_result_prob = loaded_log_reg_pipe.predict_proba(new_patient_data)





























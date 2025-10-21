# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:01:48 2025

@author: burak
"""

#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder , LabelEncoder
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.metrics import mean_squared_error , classification_report ,accuracy_score

#Load Dataset
df = pd.read_csv("Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2021_20251020.csv")
df_ = df.head(50)
df.info()
describe = df.describe()

df ["Length of Stay"] = df["Length of Stay"].replace("120 +" , 120)

df ["Length of Stay"] = pd.to_numeric(df["Length of Stay"])

los = df["Length of Stay"] 

df.isna().sum()

for column in df.columns:
    unique_values = len(df[column].unique())
    print(f"Benszersiz değer sayısı: {column}:{unique_values}")

df = df[df["Patient Disposition"] != "Expired"]

#EDA(Keşifsel Veri Analizi)
"""
yatış süresi = age - type of admission - payment type
"""
f,ax = plt.subplots()
sns.boxplot(x = "Payment Typology 1" , y = "Length of Stay" , data = df)
plt.title("Ödeme Şekli ve Yatış Arası Bağlantı")
plt.xticks(rotation = 60)
ax.set(ylim= (0,25))
plt.show()

sns.countplot(x = "Age Group" , data=df[df["Payment Typology 1"] == "Medicare" ] , order=["0 to 17" ,"18 to 29","30 to 49","50 to 69","70 or Older" ])
plt.title("Medicare Hastaları Yaşa Göre")
plt.show()

f,ax = plt.subplots()
sns.boxplot(x = "Type of Admission" , y = "Length of Stay" , data = df)
plt.title("Başvuru Sebebi ve Yatış Arası Bağlantı")
plt.xticks(rotation = 60)
ax.set(ylim= (0,25))
plt.show()

f,ax = plt.subplots()
sns.boxplot(x = "Age Group" , y = "Length of Stay" , data = df , order=["0 to 17" ,"18 to 29","30 to 49","50 to 69","70 or Older" ])
plt.title("Yaş ve Yatış Arası Bağlantı")
plt.xticks(rotation = 60)
ax.set(ylim= (0,25))
plt.show()

#Features Selection
df = df.drop(["Hospital Service Area","Hospital County","Operating Certificate Number",
              "Facility Name", "Zip Code - 3 digits" , "Patient Disposition" ,"Discharge Year","CCSR Diagnosis Description",
              "CCSR Procedure Description" ,"APR DRG Description","APR MDC Description","APR Severity of Illness Description",
              "Payment Typology 2" ,"Payment Typology 3" , "Birth Weight" ,"Total Charges","Total Costs"
              ], axis = 1)


#label encoding
age_group_index = {"0 to 17":1 ,"18 to 29":2 , "30 to 49":3 , "50 to 69":4 , "70 or Older":5}
gender_index = {"U":1 , "F":2 , "M":3}
risk_and_severity_index = {np.nan:0 , "Minor":1 , "Moderate":2 , "Major":3,"Extreme":4}

df["Age Group"] = df["Age Group"].apply(lambda x: age_group_index[x])
df["Gender"] = df["Gender"].apply(lambda x: gender_index[x])
df["APR Risk of Mortality"] = df["APR Risk of Mortality"].apply(lambda x: risk_and_severity_index[x])

encoder = OrdinalEncoder()
df["Race"] = encoder.fit_transform(np.asarray(df["Race"]).reshape(-1,1))
df["Ethnicity"] = encoder.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1,1))
df["Type of Admission"] = encoder.fit_transform(np.asarray(df["Type of Admission"]).reshape(-1,1))
df["CCSR Diagnosis Code"] = encoder.fit_transform(np.asarray(df["CCSR Diagnosis Code"]).reshape(-1,1))
df["CCSR Procedure Code"] = encoder.fit_transform(np.asarray(df["CCSR Procedure Code"]).reshape(-1,1))
df["APR Medical Surgical Description"] = encoder.fit_transform(np.asarray(df["APR Medical Surgical Description"]).reshape(-1,1))
df["Payment Typology 1"] = encoder.fit_transform(np.asarray(df["Payment Typology 1"]).reshape(-1,1))
df["Emergency Department Indicator"] = encoder.fit_transform(np.asarray(df["Emergency Department Indicator"]).reshape(-1,1))

#Tekrar Missing Value kontrolü

df = df.drop("CCSR Procedure Code" , axis = 1)

df = df.dropna(subset = ["Permanent Facility Id"])
nuller = df.isna().sum()

#train test split
X = df.drop(["Length of Stay"] , axis=1)
y = df["Length of Stay"]

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.25 , random_state=42)

#regresyon problemi
dtree = DecisionTreeRegressor(max_depth=10)
dtree.fit(X_train, y_train)
train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print("RMSE TRAIN: " , np.sqrt(mean_squared_error(y_train , train_prediction)))
print("RMSE TEST: " , np.sqrt(mean_squared_error(y_test , test_prediction)))

"""
OverFitting
RMSE TRAIN:  2.8378309828922763
RMSE TEST:  7.976411511582903

after max_depth = 10
RMSE TRAIN:  6.064167839807005
RMSE TEST:  6.221164458151911
Overfitting yok
"""



#classification problemi (yatış süresini kategorik hale getirip yapacağız)
bins  = [0,5,10,20,30,50,120]
labels = [5 , 10 , 20 ,30,50 ,120]

df["los_bin"] = pd.cut(x= df["Length of Stay"], bins=bins)
df["los_label"] = pd.cut(x= df["Length of Stay"], bins=bins , labels=labels)

df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace(","," -"))
df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace("120","120+"))

f , ax = plt.subplots()
sns.countplot(x="los_bin" ,data=df)
plt.show()

new_X = df.drop(["Length of Stay" , "los_bin","los_label"] ,axis=1)
new_y = df["los_label"]

X_train , X_test , y_train , y_test = train_test_split(new_X, new_y ,train_size=0.25 ,random_state=42)

dtree = DecisionTreeClassifier(max_depth=10)
dtree.fit(X_train, y_train)
train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print ("Train Accuracy: " ,accuracy_score(y_train, train_prediction))
print("Test Accuracy: " , accuracy_score(y_test, test_prediction))
print("Classification Report:" , classification_report(y_test , test_prediction))


"""
Parametreleri ayarlamazsan overfitting olabilir
Train Accuracy:  0.7437621169776166
Test Accuracy:  0.7397457400614346
ayarladıktan sonraki değerler
"""

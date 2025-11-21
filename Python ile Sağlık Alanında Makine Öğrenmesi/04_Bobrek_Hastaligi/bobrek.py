# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 14:44:10 2025

@author: burak
"""

#kütüphane importu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

#load dataset
df = pd.read_csv("kidney_disease.csv")
df.drop("id" , axis = 1 , inplace = True )

df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cell', 'pus_cell', 'pus_cell_clumbs', 
    'bacteria', 'blood_glucose', 'blood_urea',
       'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'packet_cell_volume', 
       'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 
       'diabetes_mellitus', 'coronary_artery_disease',
       'appetite', 'peda_edema', 'aanemia', 'class']

df.info()

describe = df.describe()

df["packet_cell_volume"] = pd.to_numeric(df["packet_cell_volume"],errors="coerce")

df["white_blood_cell_count"] = pd.to_numeric(df["white_blood_cell_count"],errors="coerce")

df["red_blood_cell_count"] = pd.to_numeric(df["red_blood_cell_count"],errors="coerce")

#EDA : KDE

cat_columns  = [col for col in df.columns if df[col].dtype == "object"]

num_columns = [col for col in df.columns if df[col].dtype != "object"]

for col in cat_columns:
    print(f"{col}.{df[col].unique()}")

df["diabetes_mellitus"].replace(to_replace = {"\tno " :"no" ,"\tyes" : "yes" , "yes" : "yes"}, inplace=True ) # inplace doğrudan kaydeder

df["coronary_artery_disease"].replace(to_replace = {"\tno " :"no" } , inplace = True )

df["class"].replace(to_replace = {"ckd\t " :"ckd" } , inplace = True )

df["class"] = df["class"].map({"ckd" : 0 , "notckd":1})

plt.figure(figsize=(15,15))
plotnumber = 1

for col in num_columns:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5 , plotnumber)
        sns.histplot(df[col] , kde = True , ax = ax)
        plt.xlabel(col)
    plotnumber += 1

plt.tight_layout()
plt.show()

#korelasyonu kontrol etmek için heatmap kullanılabilir
correlation_columns = num_columns + ["class"]
numeric_df = df[correlation_columns].copy()

plt.figure()
sns.heatmap(numeric_df.corr(), annot=True , linecolor="white" , linewidths=2)
plt.show()

def kde(col):
    grid = sns.FacetGrid(df , hue = "class" , height=6 , aspect=2)
    grid.map(sns.kdeplot,col)
    grid.add_legend()
    plt.show();
kde("hemoglobin")
kde("packet_cell_volume")
kde("white_blood_cell_count")
kde("red_blood_cell_count")
kde("specific_gravity")
kde("albumin")


#Preprocessing : missing value
df.isna().sum().sort_values(ascending = False)

def solve_mv_random_value(feature):
    
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    
    random_sample.index = df[df[feature].isnull()].index
    
    df.loc[df[feature].isnull() , feature] = random_sample
    
for col in num_columns:
    solve_mv_random_value(col)

df[num_columns].isnull().sum()

def solve_mv_mode (feature):
    
    mode = df[feature].mode()[0]
    
    df[feature] = df[feature].fillna(mode)

solve_mv_random_value("red_blood_cell")
solve_mv_random_value("pus_cell")

for col in cat_columns:
    solve_mv_mode(col)

df[cat_columns].isnull().sum()




#Preprocessing : Feature encoding

for col in cat_columns:
    print(f"{col} : {df[col].nunique()}")

encoder = LabelEncoder()

for col in cat_columns:
    df[col] = encoder.fit_transform(df[col])

#Model (DT) training

independent_col = [col for col in df.columns if col != "class" ]
dependent_col = "class"

X = df[independent_col]
y = df[dependent_col]


X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size=0.3 , random_state=42)

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print("Confusion Matrix:\n" , cm)
print("Classification Report:\n" , cr)

#DT visualization - feature importance

class_name = ["ckd" , "notckd"]

plt.figure(figsize=(20,10))

plot_tree(dtc,filled=True ,feature_names=independent_col,rounded=True , fontsize=8)
plt.show()

feature_importance = pd.DataFrame({"feature" : independent_col, "importance" :dtc.feature_importances_})

print("Most Important Feature:\n" , feature_importance.sort_values(by="importance",ascending=False).iloc[0])

plt.figure(figsize=(20,20))
sns.barplot(x="importance" , y="feature" , data=feature_importance)
plt.title("Feature Importance")
plt.show()
















































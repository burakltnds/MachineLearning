# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 18:47:00 2025

@author: burak
"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , KFold , cross_val_score , GridSearchCV
from sklearn.metrics import classification_report , confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier , GradientBoostingClassifier , RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")
 

# 1-Veri setini import et ve incele
df = pd.read_csv("diabetes.csv")

df.info()

describe = df.describe()

sns.pairplot(df , hue = "Outcome")


def plot_correlation_heatmap (df):
    
    corr_matrix = df.corr()
    
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True,fmt=".2f",linewidths=0.5,cmap="coolwarm")
    plt.title("Correlation of Features")
    #plt.show()
    
plot_correlation_heatmap(df)

# 2-Outlier Detect

def detect_outliers_iqr (df):
    
    outlier_indices = []
    outlier_df = pd.DataFrame()
    
    for col in df.select_dtypes (include=["float64" , "int64"]).columns:
        Q1 = df[col].quantile(0.25) 
        Q3 = df[col].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound ) ]
        
        outlier_indices.extend(outlier_in_col.index)
        
        outlier_df = pd.concat([outlier_df , outlier_in_col] , axis=0)
    
    #remove duplicate indices
    
    outlier_indices = list(set(outlier_indices))
    
    #remove duplicate rows in the outliers dataframe
    
    outlier_df = outlier_df.drop_duplicates()
    
    return outlier_df , outlier_indices

outlier_df , outlier_indices = detect_outliers_iqr(df)


df_cleaned = df.drop(outlier_indices).reset_index(drop=True)

# 3-train test split

X = df_cleaned.drop(["Outcome"] , axis = 1)
y = df_cleaned ["Outcome"]

X_train , X_test , y_train , y_test = train_test_split(X , y,test_size=0.25,random_state=42)


# 4-standartlaştırma

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# 5-Model Training

"""
Logistic Regresyon
Decision Tree
KNN 
GaussianNB
SVC
Adaboost , gradientboosting 
Random Forest
"""

def getBasedModel():
    basedModels = []
    basedModels.append(("LR" ,LogisticRegression()))
    basedModels.append(("DT" ,DecisionTreeClassifier()))
    basedModels.append(("KNN" ,KNeighborsClassifier()))
    basedModels.append(("NB" ,GaussianNB()))
    basedModels.append(("SVM" ,SVC()))
    basedModels.append(("AdaB" ,AdaBoostClassifier()))
    basedModels.append(("GBM" ,GradientBoostingClassifier()))
    basedModels.append(("RF" ,RandomForestClassifier()))
    
    return basedModels

def basedModelsTraning (X_train,y_train,models):
    
    results = []
    
    names = []
    
    for name , model in models:
        kfold = KFold(n_splits = 10)
        cv_results = cross_val_score(model, X_train,y_train,cv=kfold ,scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy:{cv_results.mean()} , std : {cv_results.std()}")
        
    return names , results

models = getBasedModel()

names , results = basedModelsTraning(X_train , y_train , models)

def plot_box (names , results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy")
    plt.show()

plot_box(names , results)

# 6-hyper parameter tuning

#DT

param_grid = {
    "criterion" : ["gini" , "entropy"],
    "max_depth" : [10,20,30,40,50],
    "min_samples_split" : [2,5,10] ,
    "min_samples_leaf" : [1,2,4]
    }

dt = DecisionTreeClassifier()

# 7-grid search cross validation

grid_search = GridSearchCV(estimator = dt, param_grid = param_grid , cv=5 , scoring="accuracy")

grid_search.fit(X_train , y_train)

print("En İyi Param:", grid_search.best_params_)

best_dt_model = grid_search.best_estimator_

y_pred = best_dt_model.predict(X_test)

print("Karmaşıklık Matrisi")
print(confusion_matrix (y_test , y_pred))


print("Sınıflandırma Raporu")
print(classification_report(y_test , y_pred))


# 8-Model testing with real data

new_data = np.array([[6,149,72,35,0,34.6,0.627,51]])

new_predict = best_dt_model.predict(new_data)
print("Yeni Datanın Tahmini:")
print(new_predict)












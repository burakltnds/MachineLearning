# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:26:01 2025

@author: burak
"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.model_selection import train_test_split , KFold , cross_val_score , GridSearchCV
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , VotingClassifier

import warnings
warnings.filterwarnings("ignore")

#veriyi import etme ,  inceleme
df = pd.read_csv("heart_disease_uci.csv")
df = df.drop(columns = ["id"])
df.info()
describe = df.describe()

numerical_features = df.select_dtypes (include = [np.number]).columns.tolist()

plt.figure()
sns.pairplot(df , vars=numerical_features , hue = "num")
#plt.show()
sns.countplot(x="num" , data = df)
#plt.show()

#handling missing value
print (df.isnull().sum())

df = df.drop(columns = ["ca"])
print (df.isnull().sum())

df["trestbps"].fillna ( df["trestbps"].median() , inplace = True)
df["chol"].fillna ( df["chol"].median() , inplace = True)
df["fbs"].fillna ( df["fbs"].mode()[0] , inplace = True)
df["restecg"].fillna ( df["restecg"].mode()[0] , inplace = True)
df["thalch"].fillna ( df["thalch"].median() , inplace = True)
df["exang"].fillna ( df["exang"].mode()[0] , inplace = True)
df["oldpeak"].fillna ( df["oldpeak"].median() , inplace = True)
df["slope"].fillna ( df["slope"].mode()[0] , inplace = True)
df["thal"].fillna ( df["thal"].mode()[0] , inplace = True)

print (df.isnull().sum())



#train test split
X = df.drop(["num"] , axis=1)
y = df["num"]

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.25 , random_state=42)

categorical_features = ["sex" , "dataset" , "cp" , "restecg" , "exang" , "slope" , "thal"]

numerical_features1 = ["age","trestbps","chol","fbs","thalch" ,"oldpeak"]
 
X_train_num =  X_train[numerical_features1]
X_test_num = X_test[numerical_features1]


#standartlaştırma
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

#label encoding
encoder = OneHotEncoder(sparse_output=  False , drop = "first")

X_train_cat = X_train[categorical_features]
X_test_cat = X_test [categorical_features]

X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

X_train_transformed = np.hstack((X_train_num_scaled , X_train_cat_encoded))
X_test_transformed = np.hstack((X_test_num_scaled , X_test_cat_encoded))


#modelling : rf , knn , voting classifier train ve test
rf = RandomForestClassifier(n_estimators= 100 , random_state=42)
knn = KNeighborsClassifier()

voting_clf = VotingClassifier( estimators=[("rf" , rf) , ("knn" , knn)] , voting = "soft" ) #hard olabilir

voting_clf.fit(X_train_transformed , y_train)
y_pred = voting_clf.predict(X_test_transformed)

#CM

print("Accuracy:" , accuracy_score(y_test,y_pred))
cm= confusion_matrix(y_test, y_pred)
print("CM:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True , fmt="d" , cmap="Blues",cbar=False)
plt.title("CM")
plt.xlabel("Tahmin Değeri")
plt.ylabel("Gerçek Değer")
plt.show()











































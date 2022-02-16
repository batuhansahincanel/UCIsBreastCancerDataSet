# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:17:13 2022

@author: Batuhan
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


dataset = pd.read_csv("breast_cancer.csv")

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#LogisticRegression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
#Accuracy Score & Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
ac_lr = accuracy_score(y_test, y_pred_lr)
#k-Fold Cross Validation 
acc_lr = cross_val_score(estimator = lr, X = X, y = y, cv = 10)
acc_mean_lr = acc_lr.mean()*100
std_lr = acc_lr.std()*100



#KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
#Accuracy Score & Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
ac_knn = accuracy_score(y_test, y_pred_knn)
#k-Fold Cross Validation
acc_knn = cross_val_score(estimator = knn, X = X, y = y, cv = 10)
acc_mean_knn = acc_knn.mean()*100
std_knn = acc_knn.std()*100



#SVM
from sklearn.svm import SVC
svm = SVC(probability=True, kernel="rbf")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
#Accuracy Score & Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
ac_svm = accuracy_score(y_test, y_pred_svm)
#k-Fold Cross Validation
acc_svm = cross_val_score(estimator = svm, X = X, y = y, cv = 10)
acc_mean_svm = acc_svm.mean()*100
std_svm = acc_svm.std()*100



#NaiveBayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
#Accuracy Score & Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
ac_nb = accuracy_score(y_test, y_pred_nb)
#k-Fold Cross Validation
acc_nb = cross_val_score(estimator = nb, X = X, y = y, cv = 10)
acc_mean_nb = acc_nb.mean()*100
std_nb = acc_nb.std()*100



#DecisionTree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=0, criterion="entropy")
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
#Accuracy Score & Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
ac_dt = accuracy_score(y_test, y_pred_dt)
#k-Fold Cross Validation
acc_dt = cross_val_score(estimator = dt, X = X, y = y, cv = 10)
acc_mean_dt = acc_dt.mean()*100
std_dt = acc_dt.std()*100



#RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
#Accuracy Score & Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
ac_rf = accuracy_score(y_test, y_pred_rf)
#k-Fold Cross Validation
acc_rf = cross_val_score(estimator = rf, X = X, y = y, cv = 10)
acc_mean_rf = acc_rf.mean()*100
std_rf = acc_rf.std()*100








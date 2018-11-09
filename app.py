# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:20:53 2018

@author: Skanda
"""

import pandas as pd
import dateutil.parser

def get_day_gap(my_date_difference):
    return my_date_difference.days

def get_weekday(my_date):
    return my_date.weekday()

df = pd.read_csv('KaggleV2-May-2016.csv')
df = df.iloc[0:10000]
df['day1'] = df['AppointmentDay'].apply(dateutil.parser.parse)
df['day2'] = df['ScheduledDay'].apply(dateutil.parser.parse)
df['day-gap'] = df['day1'] - df['day2']
df['day-gap'] = df['day-gap'].apply(get_day_gap)
df['AppointmentWeekDay'] = df['day1'].apply(get_weekday)
df['ScheduledWeekDay'] = df['day2'].apply(get_weekday)
df.drop('AppointmentDay', axis = 1, inplace = True)
df.drop('ScheduledDay', axis = 1, inplace = True)
df.drop('AppointmentID', axis = 1, inplace = True)
df.drop('PatientId', axis = 1, inplace = True)
df.drop('day1', axis = 1, inplace = True)
df.drop('day2', axis = 1, inplace = True)

X = pd.concat([df.iloc[:,0:9], df.iloc[:,10:]], axis = 1).values
y = df.iloc[:,9].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:, 0])
X[:,2] = labelencoder.fit_transform(X[:, 2])
y = labelencoder.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]       #Dropping a dummy column

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
pca.fit(X)
new_X = pca.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
rfc_predictions = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, rfc_predictions))
print(classification_report(y_test, rfc_predictions))

'''from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 3)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
### Best Params C = 1, gamma = 1 ##### '''

from sklearn.svm import SVC
svc = SVC(C = 1.0, gamma = 1.0)
svc.fit(X_train, y_train)
svc_predictions = svc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions))

import pickle
pickle.dump(svc, open('svm_model.txt', 'wb'))
pickle.dump(rfc, open('rfc_model.txt', 'wb'))










# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:28:54 2018

@author: kishl
"""
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Datasets and values
dataset=pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = y.reshape(len(y),1)
exp=6.5        #experience

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x= sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
expa=np.array([[exp]])
expar=sc_x.transform(expa)
# Creating the SVR model

# Creating regressor 
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

# Predicting
y_pred = regressor.predict(expar)
y_pred_descaled = sc_y.inverse_transform(y_pred)
# Visualising the SVR data
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Truth or Bluff")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

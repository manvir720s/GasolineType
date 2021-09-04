#https://intellipaat.com/blog/what-is-linear-regression/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('cars.csv', nrows=400)
dataset = dataset.drop(dataset.columns[0], axis = 1)
X = pd.DataFrame(dataset.iloc[:,[12, 13, 14]])
y = pd.DataFrame(dataset.iloc[:,1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


r_sq = regressor.fit(X_train, y_train).score(X_test, y_test) # closer to one is better 
print('coefficient of determination: ', r_sq)

print(dataset["mileage"].shape)
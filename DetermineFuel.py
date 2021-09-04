#use price, mileage, power and engine to determine whether diesel or petrol 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('Car details v4.csv', nrows = 500)
dataset = dataset.drop(dataset.columns[0], axis = 1)



# Dictionary containing the mapping
variety_mappings = {0: 'Diesel', 1: 'Petrol', 2: 'LPG', 3: 'CNG'}

# Encoding the target variables to integers
#dataset = dataset.replace(['Diesel', 'Petrol' , 'LPG', 'CNG'],[0, 1, 2, 3])

#remove units 
# (mileage = kmpl, km/kg), (engine = CC), (maxpower = bhp)\
str_factors = ["mileage", "engine", "max_power"]

for i in range(0, len(str_factors)):
    test_list = dataset[str_factors[i]]
    print(str_factors[i])
    #remove units
    if(str_factors[i] == "mileage"):
        test_list = dataset["mileage"].str.strip('kmpl')
        test_list = test_list.str.strip('km/kg')
    elif (str_factors[i] == "engine"):
        test_list = dataset["engine"].str.strip('CC')
    else:
        test_list = dataset["max_power"].str.strip('bhp')
    #convert to float
    for j in range(0, len(test_list)):
        test_list[j] = float(test_list[j])
    '''
    print(str_factors[i])
    print(test_list.mean())
    print(test_list.isnull().sum())
    print(end='\n')
    '''
    test_list.fillna(test_list.mean(), inplace = True)
    dataset[str_factors[i]] = test_list
    #price(rupees), kms driven , mileage (kmpl), engine volume (CC), horspower (bhp) 
X = pd.DataFrame(dataset.iloc[:,[1, 2, 7, 8,9]])
y = pd.DataFrame(dataset.iloc[:,3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/10, random_state = 2)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

regression = LogisticRegression()
regression.fit(X_train, y_train.values.ravel())

# Predicting the Test set results
y_pred = regression.predict(X_test)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)

#r_sq = regressor.fit(X_train, y_train.values.ravel()).score(X_test, y_test) # closer to one is better 
#rint('coefficient of determination: ', r_sq)

def classify(a, b, c, d, e):
    arr = np.array([a, b, c, d, e]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = regression.predict(query)[0] 
    return prediction # Return the prediction

print(classify(10, 100, 100, 100, 100))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
plt.rcParams['figure.figsize'] = (8.0, 4.0)

#read data and preparing data 
training_data = pd.read_csv('training.csv')
training_data['Date'] = pd.to_datetime(training_data['Date'])
training_data = training_data.set_index(training_data['Date'])
training_data = training_data.sort_index()

test_data = pd.read_csv('testing.csv')
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data = test_data.set_index(test_data['Date'])
test_data = test_data.sort_index()


# print('Train Dataset:',training_data.shape)
# print('Test Dataset:',test_data.shape)

X_train = np.array(training_data[['Date']])
X_test = np.array(test_data[['Date']])

y_train = np.array(training_data[['monthly_mean']])
y_test = np.array(test_data[['monthly_mean']])

# plt.plot(X_train, y_train)
# plt.plot(X_test, y_test)
# plt.show()

model = LinearRegression()
model.fit(X_train, y_train) 
y_predict = model.predict(y_test)


folds = KFold(n_splits = 10)
cv_predict = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=folds)


plt.plot(X_test, y_test)  # real data 
plt.plot(X_test, cv_predict) #prediction with ordinary least squares 
plt.show()


# X = data.iloc[:, 0].values.reshape(-1, 1)
# Y = data.iloc[:, 1].values.reshape(-1, 1)
# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions
# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='red')
# plt.show()


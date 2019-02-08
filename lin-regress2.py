import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv('train.csv')

df.dropna(inplace=True)

x = df.drop('x', axis=1) #split data
y = df[['x']]

df.dropna(inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

regression_model = LinearRegression()
regression_model.fit(x_train, y_train)

for idx, col_name in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))

intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


print("The score of our model is {}".format(regression_model.score(x_test, y_test)))

y_predict = regression_model.predict(x_test)

regression_model_mse = mean_squared_error(y_predict, y_test)

print('The MSE of our model is {}'.format(regression_model_mse))

math.sqrt(regression_model_mse)

print('The square root of the MSE is {}'.format(math.sqrt(regression_model_mse)))
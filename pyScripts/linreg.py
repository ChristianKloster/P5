import pandas as pd
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
import matplotlib.pyplot as plt
from matplotlib import style
import dataloader
from retailerdata import retailerData

style.use('ggplot')

# load dataframe with features:



prediction_out = 7
split_ratio = 0.8


# target value = quantity for period of prediction_out
df['target'] = [df[i:i+prediction_out]['quantity'].sum() for i in range(len(df))]

# cleaning for nan values & end of df
df = df[:-prediction_out]
df = df.dropna(how='any')

# extracting features, scaling,
X = np.array(df.drop(['target'], 1))
X = preprocessing.scale(X)

# target
y = np.array(df['target'])

#  splitting data
split_index = ceil(len(X) * split_ratio)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

df_test = df[split_index:]


# creating regressor lin_regr
reg = LinearRegression(n_jobs=-1)
# reg = Ridge(alpha = 0.5)
# reg = Lasso(alpha = 0.1)
# reg = BayesianRidge()

reg.fit(X_train, y_train)
accuracy = reg.score(X_test, y_test)

print('accuracy:')
print(accuracy)
print()
print(coefficients:)
print(reg.coef_)


# prediction_set = reg.predict(X_test)

# print(prediction_set, accuracy, prediction_out)

# df_test['Prediction'] = prediction_set


# plt.legend(loc = 4)
# plt.xlabel('Date')
# plt.ylabel('Quantity')
# # plt.show()
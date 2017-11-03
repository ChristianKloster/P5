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
retailerFilePath = 'C:/GitP5/P5/Retailers_w_coords.rpt'
transactionFilePath = 'C:/GitP5/P5/CleanData/CleanedData.rpt'
chainID = 1
style = 'J10100A'
product = 6724
daysInPeriod = 7
prediction_out = 7
lately_days = 7


# loading retailer data
df_retailers = retailerData(retailerFilePath)
df_retailers = df_retailers.get_retailers_from_chainid(chainID)
df_retailers = df_retailers['id']

# loading transaction data
df = dataloader.load_sales_file(transactionFilePath)
df = df[df.retailerID.isin(df_retailers)]

# global turnover
chaindf = df.copy()
chaindf = chaindf.groupby('date').sum()
global_turnover = chaindf['turnover']

# style
# print(df.styleNumber.value_counts())
# df = df[df.styleNumber == style]
df = df[df.productID == product]

# extracting date & quantity
df = df[['date', 'quantity']]

# group by date, converting index to datetime type
df = df.groupby('date').sum()
df.index = pd.to_datetime(df.index)

# setting up features

# global turnover
df['global_turnover'] = global_turnover

# setting up period
df['period_1'] = [df[i-daysInPeriod:i]['quantity'].sum() for i in range(len(df))]
df['period_2'] = [df[i-(daysInPeriod*2):i-daysInPeriod]['quantity'].sum() for i in range(len(df))]  # kan evt shifte  week_1 , men der sker også noget fuckery
df['period_3'] = [df[i-(daysInPeriod*3):i-(daysInPeriod*2)]['quantity'].sum() for i in range(len(df))]

# change / slope between periods
df['slope_1_2'] = (df['period_1'] - df['period_2']) / daysInPeriod
df['slope_2_3'] = (df['period_2'] - df['period_3']) / daysInPeriod

# target value = quantity for period of prediction_out
df['target'] = [df[i:i+prediction_out]['quantity'].sum() for i in range(len(df))]

# cleaning for nan values & end of df
df = df[:-prediction_out]
df = df.dropna(how='any')


predictdf = df[-lately_days:]

# extracting features, scaling,
X = np.array(df.drop(['target'], 1))

X = preprocessing.scale(X)
X_lately = X[-lately_days:]
X = X[:-lately_days]

# label
y = np.array(df['target'])
y_lately = y[-lately_days:]

y = y[:-lately_days]

# creating training set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state=666)

# creating regressor lin_regr
#reg = LinearRegression(n_jobs=-1)
reg = Ridge(alpha = 0.5)
# reg = Lasso(alpha = 0.1)
# reg = BayesianRidge()
reg.fit(X_train, y_train)
accuracy = reg.score(X_test, y_test)
accuracy2 = reg.score(X_lately, y_lately)


prediction_set = reg.predict(X_lately)

print(prediction_set, accuracy, prediction_out, accuracy2)

predictdf['Prediction']= prediction_set

df['target'].plot()
predictdf['Prediction'].plot()

plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Quantity')
# plt.show()
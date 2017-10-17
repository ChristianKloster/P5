import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoad as dl
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import average_precision_score

def modeler(d):
    # Setting up prediction columns
    columns = d.columns.tolist()
    columns = [c for c in columns if c not in ['date', 'size', 'SupplierItemgroupName', 'styleNumber', 'colorname',
                                               'isNOS', 'styleNumber', 'description']]
    # Store the variable we'll be predicting on.
    target = "quantity"
    # Generate the training set.  Set random_state to be able to replicate results.
    train = d.sample(frac=0.8, random_state=1)
    # Select anything not in the training set and put it in the testing set.
    test = d.loc[~d.index.isin(train.index)]
    # Initialize the model class.
    lin_model = LinearRegression()
    # Fit the model to the training data.
    lin_model.fit(train[columns], train[target])
    # Generate our predictions for the test set.
    lin_predictions = lin_model.predict(test[columns])
    print("Predictions:", lin_predictions)
    # Compute error between our test predictions and the actual values
    lin_mse = mean_squared_error(lin_predictions, test[target])
    print('Coefficients: \n', lin_model.coef_)
    print("Computed error:", lin_mse)
    print('Variance score: %.2f' % r2_score(test[target], lin_predictions))
    return(lin_model)

def tester():
    pass

directory = 'C:/Users/Patrick/PycharmProjects/untitled/AAU/Sales_20'
files = ['1606', '1607', '1608', '1609','1610', '1611', '1612',
         '1701', '1702', '1703', '1704', '1705', '1706' , '1707', '1708', '1709']
end = '.rpt'

for x in range(0,len(files)):
	files[x] = directory + files[x] + end

d = dl.loadSalesFiles(files)
d = d.dropna(axis=0, how='any')
d = d[d.isNOS != 1]
d = d[d.retailerID == 42]
generer_fra = 'productID'

retailers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
results = pd.DataFrame(columns=['productID', 'variance', 'error'], index=retailers)

for ret in retailers:
    a = d[generer_fra].value_counts()
    dp = d[d[generer_fra] == a.index[ret]]
    test = dp.groupby(by='date').sum()
    train = dp.groupby(by='date').sum()
    test = test.resample('W').agg({'quantity': 'sum', 'turnover': 'sum', 'discount': 'sum'})
    train = train.resample('W').agg({'quantity': 'sum', 'turnover': 'sum', 'discount': 'sum'})
    test = test.dropna(axis=0, how='any')
    train = train.dropna(axis=0, how='any')
    test = test.iloc[::-1]
    test = test.head(int(round(test.size * 0.1)))
    train = train.head(int(round(train.size * 0.9)))
    newmodel = modeler(train)
    columns = train.columns.tolist()
    columns = [c for c in columns if c not in ['date', 'size', 'SupplierItemgroupName', 'styleNumber', 'colorname',
                                               'isNOS', 'styleNumber', 'description']]
    new_pred = newmodel.predict(test[columns])
    print("Predictions clean:", new_pred)
    # Compute error between our test predictions and the actual values
    lin_mse = mean_squared_error(new_pred, test['quantity'])
    print("Computed error:", lin_mse)
    print('Variance score: %.2f' % r2_score(test['quantity'], new_pred))
    results.iloc[ret-1].set_value('productID', a.index[ret])
    results.iloc[ret-1].set_value('variance', r2_score(test['quantity'], new_pred))
    results.iloc[ret-1].set_value('error', lin_mse)
print(results)

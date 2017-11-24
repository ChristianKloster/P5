import pandas as pd
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr as pn
import matplotlib.pyplot as plt
from matplotlib import style
import dataloader
from retailerdata import retailerData

style.use('ggplot')

# load dataframe with features:

# data = ....

# select features

def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

def regress(data, features, target):
	# print(target)
	features.append(target)
	print('Features:')
	print(features)
	df = data[features].copy()

	df.dropna(inplace = True)

	split_ratio = 0.9

	# extracting features, scaling,
	X = np.array(df.drop(target, 1))
	X = preprocessing.scale(X)

	# target
	y = np.array(df[target])

	#  splitting data
	split_index = math.ceil(len(X) * split_ratio)

	X_train = X[:split_index]
	X_test = X[split_index:]

	y_train = y[:split_index]
	y_test = y[split_index:]

	df_test = df[split_index:]


	# creating regressor lin_regr
	reg = LinearRegression(n_jobs=-1)
	# reg = MLPRegressor()
	# reg = Ridge(alpha = 0.5)
	# reg = Lasso(alpha = 0.1)
	# reg = BayesianRidge()

	reg.fit(X_train, y_train)
	accuracy = reg.score(X_test, y_test)
	prediction_set = reg.predict(X_test)


	print('Model: Y = ' + pretty_print_linear(reg.coef_, features) + ' + ' + str(reg.intercept_) )
	print('accuracy (R^2) :' + str(accuracy))
	mserror = mse(y_test, prediction_set)
	print('MSE: '+ str(mserror))
	print('coefficients: ' + str(reg.coef_))
	pearson = pn(y_test, prediction_set)
	print('Pearson: ' + str(pearson))
	maxerror = max(abs(y_test - prediction_set))
	print('max error: ' + str(maxerror))
	print()

	return pd.DataFrame({'feature': features[0], 'target': features[1], 'coef':reg.coef_[0], 'intercept': reg.intercept_, 'r2': accuracy, 'mse' : mserror, 'pearson': pearson[0] , 'max_error' : maxerror}, index = [0])
	




# prediction_set = reg.predict(X_test)

# print(prediction_set, accuracy, prediction_out)

# df_test['Prediction'] = prediction_set


# plt.legend(loc = 4)
# plt.xlabel('Date')
# plt.ylabel('Quantity')
# # plt.show()
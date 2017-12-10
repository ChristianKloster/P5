
import pandas as pd
import numpy as np

from sklearn.linear_model import lasso


def learn_model(X,Y, alpha = 1.0):
	reg = lasso(alpha = alpha)
	reg.fit(X,Y)

	return reg

def pretty_print(coefs, names = None, intercept):
    res = get_significant_coefs(coefs, names)

    lst = zip(res.coef, res.name)

    sign = '-' if intercept < 0 else '+'

    return " + ".join("%s * %s" % (coef, name) for coef, name in lst) + sign + str(abs(intercept))

def get_significant_coefs(coefs, names = None):
	if names == None:
    	names = ["X%s" % x for x in range(len(coefs))]

    df = pd.DataFrame()
    df['coef'] = coefs
    df['name'] = names

    return df[df.c != 0.0]

def predict(reg, X, Y):
	return reg.predict(X,Y)

def calculate_errors(prediction, actual):
	errors = 
	{
		'rmse' : rmse(prediction, actual), 
		'mse' : mse(prediction, actual),
		'R2' : R2(prediction, actual), 
		'mae' : mae(prediction,actual),
		'max' : max(prediction,actual),
		'smape' : smape(prediction,actual),
		'max' : max(prediction,actual),
		'mape' : mape(prediction,actual)
	}

	return errors
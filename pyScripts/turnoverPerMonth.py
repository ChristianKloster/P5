import dataloader as dl
import pandas as pd
import matplotlib.pyplot as plt


def get_turnover_sum(df):
	res = df.groupby('date').sum()
	res = res['turnover']
	return res

def add_price_col(df):
	res = df.copy()
	res['price'] = (abs(res['turnover']) + abs(res['discount'])) / abs(res['quantity'])
	return res

def add_discount_PCT_col(df):
	res = df.copy()
	res['discount_PCT'] = abs(res['discount']) / (abs(res['turnover']) + abs(res['discount'])) * 100
	return res

def get_price_mean(df):
	new_df = add_price_col(df)	
	res = new_df.groupby('date').mean()
	res = res['price']
	return res


def get_discount_PCT_mean(df):
	new_df = add_discount_PCT_col(df)	
	res = new_df.groupby('date').mean()
	res = res['discount_PCT']
	return res

def plot(df):
	plt.figure()
	df.plot.bar()
	plt.show()

def main():
	path = 'c:/P5GIT/P5/GOFACT_DATA/'
	filename = 'Sales_'
	month1 = '201609'
	month2 = '201610'
	month3 = '201611'
	end = '.rpt'

	st1 = path + filename + month1 + end
	st2 = path + filename + month2 + end
	st3 = path + filename + month3 + end


	df = dl.load_sales_files([st1,st2,st3])

	data = get_discount_PCT_mean(df)

	plot(data)


if __name__ == '__main__':
	main()
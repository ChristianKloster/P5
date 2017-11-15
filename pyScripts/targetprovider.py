import pandas as pd
import dataloader as dl
import matplotlib.pyplot as plt
import numpy as np

def get_pivot_tables_with_target_values(df, retailerid = 2, days = '1W', on = 'productID'):
	data = df[df.retailerID == retailerid].copy()

	data = data[['date', on, 'quantity']]

	p = pd.pivot_table(data, values='quantity', index=['date'], columns=[on], aggfunc=np.sum).fillna(0).resample(days).sum()
	p2 = p.shift(1).fillna(0)

	return p, p2

def make_scatteplot(df, retailerid = 2, days = '1W',on = 'productID'):
	p, p2 = get_pivot_tables_with_target_values(df,retailerid, days, on)

	plt.plot(p[:],p2[:], 'b.')

	# help lines
	plt.plot([-5,30],[5,40], 'r')
	plt.plot([-5,30],[-5,30], 'g')
	plt.plot([-5,30],[-15,20], 'r')

	plt.savefig('Scatter_r_' + str(retailerid) + '_d_' + str(days) +'_' + on + '.png')


# test af kode

# # sales
# files = dl.get_all_files('C:\P5GIT\P5\GOFACT_DATA/Sales_20')
# all_df = dl.load_sales_files(files)

# all_df = all_df.reset_index()
# all_df = all_df.drop(['index'],1)


# print('data loaded')

# days = [str(i) + 'D' for i in range(1,8)]

# for i in days:
# 	make_scatteplot(all_df, 10, i, 'SupplierItemgroupName')

# make_scatteplot(all_df, 2, '1M')




# target: the qty of specific product sold next day in specific retailer

# group by retailer

# group by product
# sort by date
# shift qty 1 up

# def get_data_from_retailer(df, ID):
# 	return df[df.retailerID == ID].copy()

# def get_data_from_product(df, ID):	
# 	return df[df.productID == ID].copy()

# def make_df(date, retailerid, productid, qty, target):
# 	return pd.DataFrame({'date': date, 'retailerID' : retailerid, 'productID' : productid, 'quantity' : qty, 'target' : target}, index=[0])

# def find_target(df, date):
# 	d = date + np.timedelta64(1,'D')
# 	try:
# 		df = df.groupby('date').sum()
# 		res = df.loc[d, 'quantity']
# 	except KeyError as e:
# 		return 0.0

# def make_scatteplot(df):
# 	data = df[['date', 'retailerID', 'productID', 'quantity']].copy()

# 	# sort transactions by retailer, product and date and reset index
# 	sd = data.sort_values(['retailerID', 'productID', 'date'])
# 	sd = sd.reset_index()

# 	sd['target'] = np.nan

# 	print('data sorted')

# 	# is the next transaction for the same reatiler?
# 	sd['retailerID_s'] = sd['retailerID'] == sd['retailerID'].shift(-1)

# 	# is the next transaction for the same product?
# 	sd['productID_s'] = sd['productID'] == sd['productID'].shift(-1)

# 	# is the next transaction for the next calendar day?
# 	sd['date_s'] = sd['date'] + np.timedelta64(1,'D') == sd['date'].shift(-1)

# 	# next quantity
# 	sd['quantity_s'] = sd['quantity'].shift(-1)

# 	# if above conditions are all true set target to next quantity, else 0.0
# 	sd['target'] = tuple(map(lambda  r, p, d, q: q if r and p and d else 0.0, sd['retailerID_s'], sd['productID_s'], sd['date_s'], sd['quantity_s']))

# 	# reorder by old index
# 	result = sd[['index','target']]
# 	result = result.groupby('index').sum()

# 	data['target'] = result['target']
# 	print(data)





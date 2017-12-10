import linreg
import pandas as pd
import numpy as np

from scipy.stats import pearsonr

targets = ['target_ret_prod_rolling_7', 'target_ret_style_rolling_7', 'target_ret_prod_rolling_3', 'target_ret_style_rolling_3','target_ret_prod_rolling_1', 'target_ret_style_rolling_1']

def filter0(df):
	return df

def filter(df):
	return df[df.retailerID.isin([3])]


def filter2(df):
	return df[df.SupplierItemgroupName == 'Knitted Blouses']

def filter3(df):
	for t in targets:
		df =  df[df[t] != -9999]
	return df

# data = pd.read_csv('C:/P5GIT/featurized_ch1_all_features.csv', sep = ';', encoding = 'utf-8', parse_dates=[0])

data = pd.concat(filter2(chunck_df) for chunck_df in pd.read_csv('C:/P5GIT/featurized_ch1_all_features.csv', sep = ';', encoding = 'utf-8', parse_dates=[0], iterator=True, chunksize=10000))

# data = data[data.retailerID ==]

def get_data(filter = filter0):
	return pd.concat(filter(chunck_df) for chunck_df in pd.read_csv('C:/P5GIT/featurized_ch1_all_features.csv', sep = ';', encoding = 'utf-8', parse_dates=[0], iterator=True, chunksize=10000))

def retailer_filter(id):
	return lambda df: filter3(df[df.retailerID == id])


for t in targets:
	data = data[data[t] != -9999]
	data = data[data[t] != 0]

			# 'qty_p1_ret_prod_agg_sun',
			# 'qty_p2_ret_prod_agg_sun',
			# 'qty_p3_ret_prod_agg_sun',
			# 			'qty_p1_chain_prod_agg_sun',
			# 'qty_p2_chain_prod_agg_sun',
			# 'qty_p3_chain_prod_agg_sun',
			# 'total_quantity_chain_agg',

			# 'total_turnover_chain_agg',
features = [
			'size_scale',
			'discount_pct',
			'price',
		    'jan', 
		    'feb',
		    'mar',
		    'apr',
		    'may',
		    'jun',
		    'jul',
		    'aug',
		    'sep',
		    'oct',
		    'nov',
		    'dec',
			# 'style_age_chain',
			# 'total_turnover_chain_rolling',
			# 'total_quantity_chain_rolling',
			# 'qty_p1_chain_prod_rolling_7',
			# 'qty_p2_chain_prod_rolling_7',
			# 'qty_p3_chain_prod_rolling_7',
			# # 'avg_price_chain',
			# 'style_age_ret',
			'total_turnover_ret_rolling_7',
			'total_quantity_ret_rolling_7',
			 # 'avg_price_ret',
			'qty_p1_ret_prod_rolling_7',
			'qty_p2_ret_prod_rolling_7',
			'qty_p3_ret_prod_rolling_7',
			'qty_speed_ret_prod_p1p2_rolling_7',
    		'qty_speed_ret_prod_p2p3_rolling_7',
    		'qty_acc_ret_prod_p1p3_rolling_7', 
    		'qty_p1_ret_prod_rolling_3',
			'qty_p2_ret_prod_rolling_3',
			'qty_p3_ret_prod_rolling_3',
			'qty_speed_ret_prod_p1p2_rolling_3',
    		'qty_speed_ret_prod_p2p3_rolling_3',
    		'qty_acc_ret_prod_p1p3_rolling_3', 
    		'qty_p1_ret_prod_rolling_1',
			'qty_p2_ret_prod_rolling_1',
			'qty_p3_ret_prod_rolling_1',
			'qty_speed_ret_prod_p1p2_rolling_1',
    		'qty_speed_ret_prod_p2p3_rolling_1',
    		'qty_acc_ret_prod_p1p3_rolling_1',
    		'qty_p1_ret_style_rolling_7',
			'qty_p2_ret_style_rolling_7',
			'qty_p3_ret_style_rolling_7',
			'qty_speed_ret_style_p1p2_rolling_7',
    		'qty_speed_ret_style_p2p3_rolling_7',
    		'qty_acc_ret_style_p1p3_rolling_7', 
    		'qty_p1_ret_style_rolling_3',
			'qty_p2_ret_style_rolling_3',
			'qty_p3_ret_style_rolling_3',
			'qty_speed_ret_style_p1p2_rolling_3',
    		'qty_speed_ret_style_p2p3_rolling_3',
    		'qty_acc_ret_style_p1p3_rolling_3', 
    		'qty_p1_ret_style_rolling_1',
			'qty_p2_ret_style_rolling_1',
			'qty_p3_ret_style_rolling_1',
			'qty_speed_ret_style_p1p2_rolling_1',
    		'qty_speed_ret_style_p2p3_rolling_1',
    		'qty_acc_ret_style_p1p3_rolling_1' 
			]

allfeatures = [
'ismale',
'size_scale', 
'discount_pct', 
'price', 
'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 
'color_popularity_chain', 
'style_age_chain', 
'total_turnover_chain_rolling', 
'total_quantity_chain_rolling', 
'qty_p1_chain_prod_rolling_7', 
'qty_p2_chain_prod_rolling_7', 
'qty_p3_chain_prod_rolling_7', 
'qty_p1_chain_style_rolling_7', 
'qty_p2_chain_style_rolling_7', 
'qty_p3_chain_style_rolling_7', 
'qty_speed_chain_prod_p1p2_rolling_7', 
'qty_speed_chain_prod_p2p3_rolling_7', 
'qty_acc_chain_prod_p1p3_rolling_7', 
'qty_speed_chain_style_p1p2_rolling_7', 
'qty_speed_chain_style_p2p3_rolling_7', 
'qty_acc_chain_style_p1p3_rolling_7', 
'qty_p1_chain_prod_rolling_3', 
'qty_p2_chain_prod_rolling_3', 
'qty_p3_chain_prod_rolling_3', 
'qty_p1_chain_style_rolling_3', 
'qty_p2_chain_style_rolling_3', 
'qty_p3_chain_style_rolling_3', 
'qty_speed_chain_prod_p1p2_rolling_3', 
'qty_speed_chain_prod_p2p3_rolling_3', 
'qty_acc_chain_prod_p1p3_rolling_3', 
'qty_speed_chain_style_p1p2_rolling_3', 
'qty_speed_chain_style_p2p3_rolling_3', 
'qty_acc_chain_style_p1p3_rolling_3', 
'qty_p1_chain_prod_rolling_1', 
'qty_p2_chain_prod_rolling_1', 
'qty_p3_chain_prod_rolling_1', 
'qty_p1_chain_style_rolling_1', 
'qty_p2_chain_style_rolling_1', 
'qty_p3_chain_style_rolling_1', 
'qty_speed_chain_prod_p1p2_rolling_1', 
'qty_speed_chain_prod_p2p3_rolling_1', 
'qty_acc_chain_prod_p1p3_rolling_1', 
'qty_speed_chain_style_p1p2_rolling_1', 
'qty_speed_chain_style_p2p3_rolling_1', 
'qty_acc_chain_style_p1p3_rolling_1', 
'color_popularity_ret', 
'style_age_ret', 
'total_turnover_ret_rolling_7', 
'total_quantity_ret_rolling_7', 
'qty_p1_ret_prod_rolling_7', 
'qty_p2_ret_prod_rolling_7', 
'qty_p3_ret_prod_rolling_7', 
'qty_speed_ret_prod_p1p2_rolling_7', 
'qty_speed_ret_prod_p2p3_rolling_7', 
'qty_acc_ret_prod_p1p3_rolling_7', 
'qty_p1_ret_prod_rolling_3', 
'qty_p2_ret_prod_rolling_3', 
'qty_p3_ret_prod_rolling_3', 
'qty_speed_ret_prod_p1p2_rolling_3', 
'qty_speed_ret_prod_p2p3_rolling_3', 
'qty_acc_ret_prod_p1p3_rolling_3', 
'qty_p1_ret_prod_rolling_1', 
'qty_p2_ret_prod_rolling_1', 
'qty_p3_ret_prod_rolling_1', 
'qty_speed_ret_prod_p1p2_rolling_1', 
'qty_speed_ret_prod_p2p3_rolling_1', 
'qty_acc_ret_prod_p1p3_rolling_1', 
'qty_p1_ret_style_rolling_7', 
'qty_p2_ret_style_rolling_7', 
'qty_p3_ret_style_rolling_7', 
'qty_speed_ret_style_p1p2_rolling_7', 
'qty_speed_ret_style_p2p3_rolling_7', 
'qty_acc_ret_style_p1p3_rolling_7', 
'qty_p1_ret_style_rolling_3', 
'qty_p2_ret_style_rolling_3', 
'qty_p3_ret_style_rolling_3', 
'qty_speed_ret_style_p1p2_rolling_3', 
'qty_speed_ret_style_p2p3_rolling_3', 
'qty_acc_ret_style_p1p3_rolling_3', 
'qty_p1_ret_style_rolling_1', 
'qty_p2_ret_style_rolling_1', 
'qty_p3_ret_style_rolling_1', 
'qty_speed_ret_style_p1p2_rolling_1', 
'qty_speed_ret_style_p2p3_rolling_1', 
'qty_acc_ret_style_p1p3_rolling_1'
]


def univariate():
	df = pd.DataFrame()

	for t in targets:
		for f in features:
			s = linreg.regress(data, [f], t)
			df = df.append(s)


	for t in targets:
		temp = df[df.target == t]
		print(t + ':')
		print(temp.groupby(['feature']).mean())

	print('mean')
	print(df.groupby(['target']).mean())
	print('max')

	print(df.groupby(['target']).max())
	print('min')
	print(df.groupby(['target']).min())

	df.to_csv('stats_ch1_PERIOD_features.csv', na_rep = 'bananaphone', index = False, sep=';',encoding='utf-8')

def model_tester(target, F = []):
	df = pd.DataFrame()

	for f in features:
		if f in F:
			print('skipped: ' + f)
			continue
		FE = F.copy().append(f) if F else [f]
		s = linreg.regress(data, FE, target)
		df = df.append(s)

	res = df.groupby(['feature']).mean()
	res.to_csv('C:/P5GIT/eksperiment_'+ target + '_iter_' + str(len(F)) + '.csv', na_rep = 'bananaphone', index = True, sep=';',encoding='utf-8')
	print(res)
	

	#df.to_csv('stats_ch1_PERIOD_features.csv', na_rep = 'bananaphone', index = False, sep=';',encoding='utf-8')

def model(df, myfeatures, target, alp):

	print(target)
	print()
	s = linreg.lasso(df, myfeatures.copy(), target, alp)
	print(s)
	print()
	print()
	return s



# model_tester('target_ret_prod_rolling_7', [])

rets = [3,4]

for r in rets:
	data = get_data(retailer_filter(r))

	results = pd.DataFrame()

	for t in targets:
		for alpha in [1.0, 0.5, 0.1]:
			print('alpha = ' + str(alpha))
			s = model(data, allfeatures, t, alpha)
			results = results.append(s)

grps = results.groupby(['target', 'alpha'])

for i,grp in grps:
	print(i)
	print(grp.mean())


results.to_csv('C:/P5GIT/eksperiment_all.csv', na_rep = 'bananaphone', index = True, sep=';',encoding='utf-8')


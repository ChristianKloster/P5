import linreg
import pandas as pd

from scipy.stats import pearsonr

data = pd.read_csv('C:/P5GIT/featurized_ch1_new.csv', sep = ';', encoding = 'utf-8')

target = 'target_prod_rolling_7'

data = data[data[target] != -9999]

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
			'style_age_chain',
			'total_turnover_chain_rolling',
			'total_quantity_chain_rolling',
			'qty_p1_chain_prod_rolling_7',
			'qty_p2_chain_prod_rolling_7',
			'qty_p3_chain_prod_rolling_7',
			# 'avg_price_chain',
			'style_age_ret',
			'total_turnover_ret_rolling_7',
			'total_quantity_ret_rolling_7',
			# 'avg_price_ret',
			'qty_p1_ret_prod_rolling_7',
			'qty_p2_ret_prod_rolling_7',
			'qty_p3_ret_prod_rolling_7',
			'qty_speed_ret_prod_p1p2',
    		'qty_speed_ret_prod_p2p3',
    		'qty_acc_ret_prod_p1p3' 
			]


def univariate():
	df = pd.DataFrame()

	for f in features:
		s = linreg.regress(data, [f], target)
		df = df.append(s)

	print(df.groupby(['feature']).mean())

	df.to_csv('stats_ch1_features.csv', na_rep = 'bananaphone', index = False, sep=';',encoding='utf-8')

def model(myfeatures):

	s = linreg.regress(data, myfeatures, target)


testfeatures = [
			# 'size_scale',
			# 'discount_pct',
			# 'price',
			# 'style_age_chain',
			# 'total_turnover_chain_rolling',
			# 'total_quantity_chain_rolling',
			# 'qty_p1_chain_prod_rolling_7',
			# 'qty_p2_chain_prod_rolling_7',
			# 'qty_p3_chain_prod_rolling_7',
			# 'avg_price_chain',
			# 'style_age_ret',
			# 'total_turnover_ret_rolling_7',
			# 'total_quantity_ret_rolling_7',
			# 'avg_price_ret',
			'qty_p1_ret_prod_rolling_7',
			# 'qty_p2_ret_prod_rolling_7',
			# 'qty_p3_ret_prod_rolling_7',
			'qty_speed_ret_prod_p1p2',
    		# 'qty_speed_ret_prod_p2p3',
    		'qty_acc_ret_prod_p1p3' 
			]

model(testfeatures)


import linreg
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

# style.use('ggplot')


data = pd.read_csv('featurized_ch1_new.csv', sep = ';', encoding = 'utf-8')



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


target = 'target_prod_rolling_7'

data = data[data[target] != -9999]

df = pd.DataFrame()

for f in features:

	plt.Figure()
	plt.plot(data[f], data[target], 'b.')
	plt.ylabel(target)
	plt.xlabel(f)
	plt.savefig(f +'_'+ target + '.png')
	plt.close()


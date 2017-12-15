import pandas as pd


ch = pd.read_csv('C:/P5GIT/featurized_ch1_all_chain_features.csv',sep=';',encoding='utf-8')
rt = pd.read_csv('C:/P5GIT/featurized_ch1_all_ret_features.csv',sep=';',encoding='utf-8')


features = [
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
    		'qty_acc_ret_style_p1p3_rolling_1',
			'target_ret_prod_rolling_7', 
			'target_ret_style_rolling_7',
			'target_ret_prod_rolling_3', 
			'target_ret_style_rolling_3',
			'target_ret_prod_rolling_1', 
			'target_ret_style_rolling_1'
			]

for f in features:
	ch[f] = rt[f]

ch.to_csv('C:/P5GIT/featurized_ch1_all_features.csv', na_rep = 'bananaphone', index = False, sep=';',encoding='utf-8')

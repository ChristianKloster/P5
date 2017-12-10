import linreg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

# style.use('ggplot')


data = pd.read_csv('C:/P5GIT/featurized_ch1_period_testing.csv', sep = ';', encoding = 'utf-8')



targets = ['target_prod_rolling_7', 'target_style_rolling_7','target_prod_rolling_3', 'target_style_rolling_3','target_prod_rolling_1', 'target_style_rolling_1']

for t in targets:
	data = data[data[t] != -9999]

for t in targets:
	df = data[t].value_counts()
	df = df.sort_index()

	plt.Figure()
	# plt.yscale('log')
	plt.bar(df.index, (df.values))
	plt.xlabel(t)
	plt.ylabel('antal')

	plt.savefig('C:/P5GIT/' + 'histogram_normal'+'_'+ t + '.png')
	plt.close()
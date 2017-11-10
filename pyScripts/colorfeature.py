import dataloader as dl
import pandas as pd
import retailerdata as rd
import numpy as np

class ColorFeatureProvider:
	def __init__(self, df, window = '7d'):
		df = df.copy()
		df.colorname = df.colorname.str.lower().str.strip()
		df_table = pd.pivot_table(df, values='quantity', index=['date'], columns=['colorname'], aggfunc=np.sum).fillna(0)
		df_table = df_table.rolling(window).mean()
		self.table = df_table

	def get_color_popularity(self, color=None, date=None):
		if color == None and date == None:
			return self.table
		elif color == None:
			return self.table.loc[date,:]
		elif date == None:
			return self.table.loc[:,color]
		else:
			return self.table.loc[date, color]

def make_feature_col(df, window = '7d'):
	cfp = ColorFeatureProvider(df, window)
	print('table made')
	data = df.copy()
	data.colorname = data.colorname.str.lower().str.strip()
	data = data.dropna(axis=0, how='any')

	col_name = 'color_popularity'
	data[col_name] = np.nan

	data[col_name] = tuple(map(lambda color, date: cfp.get_color_popularity(color = color, date = date), data['colorname'], data['date']))

	return data



# test af kode

# sales
files = dl.get_all_files('C:\P5GIT\P5\GOFACT_DATA/Sales_20')
all_df = dl.load_sales_files(files)

# retailers
RD = rd.retailerData('C:\P5GIT\P5\GOFACT_DATA/Retailers_w_coords.rpt')
retailer_data = RD.get_dataframe()

# kæde 1
ch1 = retailer_data[retailer_data.chainid == 1]
ch1_retailers = ch1['id']
ch1_sales = all_df[all_df.retailerID.isin(ch1_retailers)].copy()


# instans:

new_df = make_feature_col(ch1_sales)

print(new_df)

# cfp = ColorFeatureProvider(ch1_sales)

# eksempler på brug
# table1 =  cfp.get_color_popularity(date = pd.date_range('2017-01-01', '2017-01-15').tolist())
# table2 =  cfp.get_color_popularity(color = ['black', 'rose dust'])
# table3 =  cfp.get_color_popularity(color = 'black', date = '2016-12-17')
# table4 =  cfp.get_color_popularity(date = '2016-12-17')
# table5 =  cfp.get_color_popularity(date = '2017-09-10')
# table6 =  cfp.get_color_popularity()

# print(table5)
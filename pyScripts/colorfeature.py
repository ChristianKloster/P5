import dataloader as dl
import pandas as pd
import retailerdata as rd
import numpy as np

class ColorFeatureProvider:
	def __init__(self, df, window = '7d'):
		df = df.copy()
		df.loc[:,'colorname'] = df.loc[:,'colorname'].str.lower().str.strip()
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


# instans
cfp = ColorFeatureProvider(ch1_sales)

# table1 =  cfp.get_color_popularity(date = pd.date_range('2017-01-01', '2017-01-15').tolist())
table2 =  cfp.get_color_popularity(color = ['black', 'rose dust'])
# table3 =  cfp.get_color_popularity(color = 'black', date = '2016-12-17')
# table4 =  cfp.get_color_popularity(date = '2016-12-17')
# table5 =  cfp.get_color_popularity(date = '2016-12-17')
# table6 =  cfp.get_color_popularity()

# print(table1)
# print(table2)
# print(table3)
# print(table4)
# print(table5)
# print(table6)

import matplotlib.pyplot as plt

plt.figure()
table2.plot()
plt.show()

# table.to_csv('ch1_colors.csv')









# ch1_colors = ch1_sales.colorname.unique()
# ch2_colors = ch2_sales.colorname.unique()
# ch3_colors = ch3_sales.colorname.unique()

# ch1_colors = pd.Series(ch1_colors).str.lower().str.strip().dropna()
# ch2_colors = pd.Series(ch2_colors).str.lower().str.strip().dropna()
# ch3_colors = pd.Series(ch3_colors).str.lower().str.strip().dropna()

# ch1_colors = ch1_colors.unique()
# ch2_colors = ch2_colors.unique()
# ch3_colors = ch3_colors.unique()

# ch1_colors =  sorted(ch1_colors)
# ch2_colors =  sorted(ch2_colors)
# ch3_colors =  sorted(ch3_colors)

# # print(ch1_colors, (ch2_colors),(ch3_colors))

# print(len(ch1_colors), len(ch2_colors),len(ch3_colors))


# colorname_dict = {'blacks':['black', 'night sky'], 
# 					'whites':['white', 'vanilla']}

# color_seperated = {}

# for colorkey in colorname_dict:
# 	cs = pd.Series(ch1_colors)
# 	s = pd.Series()

# 	for color in colorname_dict[colorkey]:
# 		# get alle colorNames containing c form myColors
# 		s = s.append(cs[cs.str.contains(color)])
# 		# put list in dictionary
# 	color_seperated[colorkey] = s
# 		# remove colors from series
# 		# cs = cs.drop(s.keys())

# print(color_seperated)


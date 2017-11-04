import dataloader as dl
import pandas as pd
import retailerdata as rd
import numpy as np


files = dl.get_all_files('C:\P5GIT\P5\GOFACT_DATA/Sales_20')
all_df = dl.load_sales_files(files)

RD = rd.retailerData('C:\P5GIT\P5\GOFACT_DATA/Retailers_w_coords.rpt')

retailer_data = RD.get_dataframe()

ch1 = retailer_data[retailer_data.chainid == 1]
ch2 = retailer_data[retailer_data.chainid == 2]
ch3 = retailer_data[retailer_data.chainid == 4]

ch1_retailers = ch1['id']
ch2_retailers = ch2['id']
ch3_retailers = ch3['id']

ch1_sales = all_df[all_df.retailerID.isin(ch1_retailers)]
ch2_sales = all_df[all_df.retailerID.isin(ch2_retailers)]
ch3_sales = all_df[all_df.retailerID.isin(ch3_retailers)]

ch1_colors = ch1_sales.colorname.unique()
ch2_colors = ch2_sales.colorname.unique()
ch3_colors = ch3_sales.colorname.unique()

ch1_colors = pd.Series(ch1_colors).str.lower().str.strip().dropna()
ch2_colors = pd.Series(ch2_colors).str.lower().str.strip().dropna()
ch3_colors = pd.Series(ch3_colors).str.lower().str.strip().dropna()

ch1_colors = ch1_colors.unique()
ch2_colors = ch2_colors.unique()
ch3_colors = ch3_colors.unique()

ch1_colors =  sorted(ch1_colors)
ch2_colors =  sorted(ch2_colors)
ch3_colors =  sorted(ch3_colors)

# print(ch1_colors, (ch2_colors),(ch3_colors))

print(len(ch1_colors), len(ch2_colors),len(ch3_colors))


colorname_dict = {'blacks':['black', 'night sky'], 
					'whites':['white', 'vanilla']}

color_seperated = {}

for colorkey in colorname_dict:
	cs = pd.Series(ch1_colors)
	s = pd.Series()

	for color in colorname_dict[colorkey]:
		# get alle colorNames containing c form myColors
		s = s.append(cs[cs.str.contains(color)])
		# put list in dictionary
	color_seperated[colorkey] = s
		# remove colors from series
		# cs = cs.drop(s.keys())

print(color_seperated)


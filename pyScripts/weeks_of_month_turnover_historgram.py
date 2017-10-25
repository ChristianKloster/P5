import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from collections import OrderedDict
from retailerdata import retailerData as rd


def filter_data(df):
	d = df.dropna(axis=0, how='any')
	d = d[d.isNOS != 1]	#Ignorer NOS varer
	d = d[(d.date.dt.month != 12) & (d.date.dt.month != 1)]	#ignorer December og Januar

	rddf = rd('C:/Users/SM-Baerbar/Documents/GitHub/P5/Retailers_w_coords.rpt')
	df_danish = rddf.get_retailers_from_country('Denmark')
	d = d[d.retailerID.isin(df_danish.id)]  # Kun danske butikker
	return d

def between_days_sum(start_date, end_date):
	d = df[(df.date.dt.day >= start_date) & (df.date.dt.day <=end_date)]
	return d.turnover.sum()

#Metode der modtager laver et diagram som viser fordelingen af salget over farver og størrelser hos en enkelt vare.
#Metoden modtager et input data og et navn. Dataen skal også indeholde "row keys" som bliver farven på søjlen.
def histogram(inputdata, styleName):
	df = pd.DataFrame(OrderedDict(inputdata),index=pd.Index([''], name=styleName))
	colors = ['black', 'white', 'blue', 'red']
	x = df.plot(kind='bar', color=(colors), edgecolor ='black')
	x.set_ylabel('Turnover')
	plt.tight_layout()
	plt.show()

df = dl.load_sales_files_ranges('C:/Users/SM-Baerbar/Documents/GitHub/P5/GOFACT_DATA/Sales_20', 1606, 1613, 1701, 1710)
df = filter_data(df)

raw_data ={
	'days 1-8':    [(between_days_sum(1, 8)/8)],
	'days 8-15':    [(between_days_sum(8, 15)/8)],
	'days 16-23':     [(between_days_sum(16, 23)/8)],
	'days 24-31':      [(between_days_sum(24, 31)/7.416)]
}

print(raw_data)

histogram(raw_data,'Time')
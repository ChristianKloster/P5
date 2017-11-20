import gmplot as gm
import geoCoords as gc
import pandas as pd
import retailerdata as rd
import random as rnd
import DataLoading as dl


jitter = 0.015

RD = rd.retailerData('Retailers_w_coords.rpt')

def get_color_from_chain(retailerid):
	ch = RD.get_chainnumber_from_retailerid(retailerid)
	if ch == 1:
		return 'r'
	elif ch == 2:
		return 'g'
	elif ch == 4:
		return 'b'
	else:
		return 'k'

def get_color_from_region(retailerid):
	ch = RD.get_region_from_retailerid(retailerid)
	if ch == 'Nordjylland':
		return 'r'
	elif ch == 'Midtjylland':
		return 'g'
	elif ch == 'Sønderjylland':
		return 'b'
	elif ch == 'Fyn':
		return 'c'
	elif ch == 'Sjælland':
		return 'm'
	elif ch == 'Udland':
		return 'y'
	else:
		return 'k'

df = RD.get_dataframe()
# df = RD.get_retailers_from_region('Midtjylland')
#df = df[df.country == 'Denmark']
df = df[df.chainid == 4]

lats = df['lat'].get_values()
lngs = df['lng'].get_values()

# sales = dl.loadAllSales()

# sales = sales.groupby('retailerID').sum()

# df = df[df.id.isin(sales[0])]

# turnover = sales['turnover']
# maxturnover = turnover.max()
# minturnover = turnover.min()

# normalized_turnover = (turnover - minturnover)/(maxturnover - minturnover) * 100

# for i in range(0,len(df)):
# 	row = df.iloc[i]
# 	retailerID = row['id']
# 	trn = 0
# 	if normalized_turnover.index.
# 		trn = normalized_turnover[retailerID]
# 	while trn > 0:
# 		lats.append(row['lat'])
# 		lngs.append(row['lng'])
# 		trn -= 1


gmap=gm.GoogleMapPlotter(lats[0],lngs[0], 7)

#gmap.heatmap(lats,lngs, radius = 25)

for i in range(0,len(df)):
	row = df.iloc[i]
	retailerID = row['id']
	lat = row['lat'] + (rnd.random() - 0.5) * jitter
	lng = row['lng'] + (rnd.random() - 0.5) * jitter
	gmap.marker(lat, lng, c = get_color_from_region(retailerID), title = str(retailerID))
	# gmap.circle(lat, lng, c = get_color_from_chain(retailerID), radius = 10000)


gmap.draw('ch3_all_map.html')

# coords = []
# lat=[]
# lng=[]

# for i in range(0,len(df)):
# 	row = df.iloc[i]
# 	coords.append((row[2],gc.getCoords(row[6], row[7])))

# for field in coords:
# 	ID = field[0]
# 	lat = field[1]['lat']
# 	lng = field[1]['lng']
# 	gmap.marker(lat, lng,title=str(ID))


# df['lat'] = lat
# df['lng'] = lng

# df.to_csv('Retailers_w_coords.rpt' ,index=False ,sep=';')
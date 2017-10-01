import requests as rq
import pandas as pd
import numpy as np
import time

apiservice = 'http://api.gofact.net/api/external/stock/retailerid/'
apikey = 'c2eef1a1-9791-4974-bc16-1b18a2224f38'

# All retailers present in 'Retailers.rpt'
retailers = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14,15,16,17,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46, 47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,88,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,166,167,168,170,180,181,182,183,185,247,248,250,332,333,349,360,362,363,364,418,421,436,451,455,459,460,466,469,471,472,474,475]

# Get stock from all retailers and save in rpt file 'Stock_{date}.rpt'
def main():
	date = time.strftime("%Y%m%d")
	df = pd.DataFrame()

	for ret in retailers:
		data = getStock(ret)
		row = pd.DataFrame(data)
		df = df.append(row)

	df.to_csv('Stock_' + date + '.rpt' ,index=False ,sep=';')


# Make request for stock data. Returns list of dicts '[{productId:val, retailerId:val, stockcount:val },...]'
def getStock(retailerID):
	s = makeRequestString(retailerID)
	headers = {'Authorization' : 'gofact_token ' + apikey, 'Content-Type':'application/json'}
	response = rq.get(s, headers=headers)
	data = response.json()
	return data

def makeRequestString(retailerID):
	return apiservice + str(retailerID)

if __name__ == '__main__':
	main()
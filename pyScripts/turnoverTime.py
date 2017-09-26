import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadSalesFiles(listOfSalesFiles):
	df = pd.DataFrame()
	for file in listOfSalesFiles:
		df = df.append(loadSalesFile(file))
	return df

def loadSalesFile(filePath):
	return pd.read_csv(filePath, encoding='utf-8',
								sep=';', 
								parse_dates=[0], # parse first column
								skiprows=1, # skip header
								comment='(', # ignore the '(n rows affected)' footer
								names=[
									'date', 
									'supplierID', 
									'retailerID', 
									'supplierItemgroupID', 
									'SupplierItemgroupName', 
									'productID',
									'styleNumber',
									'description',
									'colorname',
									'isNOS',
									'size',
									'quantity',
									'turnover',
									'discount' ],
								dtype={
									'supplierID':np.int64, 
									'retialerID':np.int64, 
									'supplierItemgroupID':np.int64, 
									'SupplierItemgroupName':np.str_, 
									'productID':np.int64,
									'styleNumber':np.str_,
									'description':np.str_,
									'colorname':np.str_,
									'isNOS':np.int64,
									'size':np.str_,
									'quantity':np.int64,
									'turnover':np.float64,
									'discount':np.float64})

# ad hoc testing:
#C:\Users\SorenM\Documents\GitHub\P5\GOFACT_DATA
directory = 'C:/Users/SorenM/Documents/GitHub/P5/GOFACT_DATA/Sales_20'
files = []
for r in range(1606,1612):
	files.append('%04i' % r)

for r in range(1701,1709):
	files.append('%04i' % r)

end = '.rpt'

for x in range(0,len(files)):
	files[x] = directory + files[x] + end

d = loadSalesFiles(files)

col_list = ["turnover", "date", "productID"]
d = d[col_list]
x=d["date"]
y=d["turnover"]
#d= d[(d.turnover > 5) & (d.turnover < 0)]
#print(d)
#print(d['description'].value_counts())

N=10000
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

d=pd.DataFrame(runningMeanFast(y,N))
y=d

plt.plot(x,d)
plt.show()
#plt.savefig('%s.png' %"test")
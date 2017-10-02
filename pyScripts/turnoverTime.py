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

#Returns the moving average of x with the average made over N points
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

#Returns the dataframe containing all sales information from periods ym1_1 to ym1_2 and ym2_1 to ym2_2
# (ym stands for year/month)
def createDataframe(filepath, ym1_1, ym1_2, ym2_1, ym2_2):
	df = pd.DataFrame()
	files = []
	for r in range(ym1_1, ym1_2):
		files.append('%04i' % r)

	for r in range(ym2_1, ym2_2):
		files.append('%04i' % r)

	end = '.rpt'

	for x in range(0, len(files)):
		files[x] = filepath + files[x] + end
	df = loadSalesFiles(files)
	return df

#Returns a DF containing only relevant collumns
def col_listOfDF(filepath, ym1_1, ym1_2, ym2_1, ym2_2):
	df=createDataframe(filepath, ym1_1, ym1_2, ym2_1, ym2_2)
	col_list = ["turnover", "date"]
	df = df[col_list]
	return df

#Creates a plot of the turnover over a given period of time on which
# the moving average (of x with the average made over N points) was performed on
def createPlot_TurnoverOverTime(filepath, ym1_1, ym1_2, ym2_1, ym2_2, N):
	df=col_listOfDF(filepath, ym1_1, ym1_2, ym2_1, ym2_2)
	x=df["date"]
	y=df["turnover"]
	#d= d[(d.turnover > 5) & (d.turnover < 0)]
	#print(d)
	#print(d['description'].value_counts())

	newY=pd.DataFrame(runningMeanFast(y, N))		#Moving average of turnover

	plt.plot(x,newY)

N=20000
createPlot_TurnoverOverTime('C:/Users/SM-Baerbar/Documents/AAU/Semester 5/P5/P5/GOFACT_DATA/Sales_20', 1606, 1613, 1701,
							1710, N)
plt.show()
#plt.savefig('%s.png' %"test")



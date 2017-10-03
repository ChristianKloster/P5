import pandas as pd
import numpy as np

# returns single dataframe containing data from multiple files
def load_sales_files(listOfSalesFiles):
	df = pd.DataFrame()
	for file in listOfSalesFiles:
		df = df.append(loadSalesFile(file))
	return df

# dataframe from singe file with type checking and naming
def load_salesfile(filePath):
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

# directory = 'C:/Users/nicol/Desktop/AAU/Sales_20'
# files = ['1608', '1609','1610']
# end = '.rpt'

# for x in range(0,len(files)):
# 	files[x] = directory + files[x] + end

# d = loadSalesFiles(files)

# print(d['description'].value_counts())

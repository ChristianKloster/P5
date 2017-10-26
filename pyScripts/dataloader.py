import pandas as pd
import numpy as np

# returns single dataframe containing data from multiple files
def load_sales_files(listOfSalesFiles):
	df = pd.DataFrame()
	for file in listOfSalesFiles:
		df = df.append(load_sales_file(file))
	return df

# dataframe from singe file with type checking and naming
def load_sales_file(filePath):
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

# Returns the dataframe containing all sales information from periods ym1_1 to ym1_2 and ym2_1 to ym2_2
# (ym stands for year/month)
def load_sales_files_ranges(filepath, ym1_1, ym1_2, ym2_1, ym2_2):
    files = []
    for r in range(ym1_1, ym1_2):
        files.append('%04i' % r)

    for r in range(ym2_1, ym2_2):
        files.append('%04i' % r)

    end = '.rpt'

    for x in range(0, len(files)):
        files[x] = filepath + files[x] + end
    df = load_sales_files(files)
    return df

def get_columns(df, columns):
    df = df[columns]
    return df

files = ['1606', '1607', '1608', '1609','1610', '1611', '1612',
         '1701', '1702', '1703', '1704', '1705', '1706' , '1707', '1708', '1709']
end = '.rpt'

def get_all_files(direc):
	output = []
	for file in files:
		output.append(direc + file + end)
	return output

# ad hoc testing:

# directory = 'C:/Users/nicol/Desktop/AAU/Sales_20'
# files = ['1608', '1609','1610']
# end = '.rpt'

# for x in range(0,len(files)):
# 	files[x] = directory + files[x] + end

# d = loadSalesFiles(files)

# print(d['description'].value_counts())

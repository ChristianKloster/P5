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
									'discount',
									'chainID',
									'ismale'],
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
									'discount':np.float64,
									'chainID':np.int64,
									'ismale':np.int64})


def load_feature_file(filePath):
	return pd.read_csv(filePath, encoding='utf-8',
								sep=';',
								parse_dates=[0], # parse first column
								skiprows=1, # skip header
								comment='(', # ignore the '(n rows affected)' footer
								names=[
	                                'mon',
	                                'tue',
	                                'wed',
	                                'thu',
	                                'fri',
	                                'sat',
	                                'sun',
									'quantityChain',
									'turnoverChain',
									'jan',
									'feb',
									'mar',
									'apr',
									'may',
									'jun',
									'jul',
									'aug',
									'sep',
									'okt',
									'nov',
									'dec',
									'month_change_dist',
									'month_first_dist',
									'month_next_first_dist',
									'quantity7d',
									'quantity30d',
									'discount_percent',
									'avg_price',
									'avg_price_style',
									'color',
									'lifetimeChain',
									'lifetimeRetailer',
									'accelerationChainStyle',
									'accelerationRetailerStyle',
									'PLCBD',
									'PLCCD',
									'quantityPeriod1',
									'quantityPeriod2',
									'p1-p2',
									'quantityPeriod3',
									'p2-p3'],
								dtype={
									'mon':np.int64,
									'tue':np.int64,
									'wed':np.int64,
									'thu':np.int64,
									'fri':np.int64,
									'sat':np.int64,
									'sun':np.int64,
									'quantityChain': np.int64,
									'turnoverChain': np.float64,
									'jan':np.int64,
									'feb':np.int64,
									'mar':np.int64,
									'apr':np.int64,
									'may':np.int64,
									'jun':np.int64,
									'jul':np.int64,
									'aug':np.int64,
									'sep':np.int64,
									'okt':np.int64,
									'nov':np.int64,
									'dec':np.int64,
									'month_change_dist':np.int64,
									'month_first_dist':np.int64,
									'month_next_first_dist':np.int64,
									'quantity7d':np.int64,
									'quantity30d':np.int64,
									'discount_percent':np.float64,
									'avg_price':np.float64,
									'avg_price_style':np.float64,
									'color':np.float64,
									'lifetimeChain':np.int64,
									'lifetimeRetailer':np.int64,
									'accelerationChainStyle':np.float64,
									'accelerationRetailerStyle':np.float64,
									'PLCBD':np.float64,
									'PLCCD':np.float64,
									'quantityPeriod1':np.int64,
				                    'quantityPeriod2':np.int64,
									'p1-p2':np.float64,
									'quantityPeriod3':np.int64,
									'p2-p3':np.float64})



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

import pandas as pd
import numpy as np

directory = 'C:/git repo/P5/GOFACT_DATA/Sales_201612.rpt'

banned_Words = ['Dummy','MEN - BASIC - DON\'T USE','WOMEN - BASIC - DON\'T USE'] #Tilføj ord?

#Dataloader
df = pd.read_csv(directory, encoding='utf-8',
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

for word in banned_Words:
    df = df[df.SupplierItemgroupName != word]

df.to_csv(path_or_buf=r'C:\Users\TheChamp\Desktop\out\file.txt') #To ensure they were removed from the dataframe

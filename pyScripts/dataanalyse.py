import dataloader as dl
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import retailerdata as rd

dir_patrick = 'C:/Users/Patrick/PycharmProjects/untitled/CleanData/CleanedData.rpt'
dir_SM = r'C:\Users\SMSpin\Documents\GitHub\P5\CleanData\CleanedData_no_isnos.rpt'

df = dl.load_sales_file(dir_SM)

columns = df.columns.tolist()
columns = [c for c in columns if c not in ['quantity', 'date', 'SupplierItemgroupName', 'size', 'styleNumber', 'colorname', 'description']]

# columns = [c for c in columns if c in ['discount', 'turnover']]
print(df.describe())
for c in columns:
    y = df['quantity']
    x = df['{0}'.format(c)]#/df['quantity']
    plt.close()
    plt.figure()
    plt.plot(x, y, '.')
    plt.ylabel('quantity')
    plt.xlabel('{0}'.format(c))
    plt.title('quantity vs {0}'.format(c))
    plt.tight_layout()
    plt.savefig('quantity_vs_{0}.png'.format(c))
    print('{0} DONE'.format(c))
import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from collections import OrderedDict
from calendar import monthrange
from retailerdata import retailerData as rd
import os
import sys
import time
import numpy as np
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

def weekFeature(df, retailerID):


def SoerenSort(df, strdate, retailerID):
    #date = df.date.strptime(strdate, '%d%b%Y')

    df = df[df.retailerID == retailerID]

    #Df containing only required columns with date as index
    col_list = ["date", "quantity"]
    df = df[col_list]
    df.set_index('date', inplace=True)
    #df.index = df['date']
    #df = df.drop('date', 1)


    weeks_sum_list = df.resample('W').sum()
    print(weeks_sum_list)
    print(df)


    df.quantity = np.nan
    print(df)

    print(df.iloc[0][0])
    print(weeks_sum_list.iloc[0][0])
    print(df.index.values[0])
    print(weeks_sum_list.index.values[0])
    if df.index.values[0] == weeks_sum_list.index.values[0]:
        print("Same date!")

    for value in weeks_sum_list.index.values:
        if value in df.index:
            df.at[value, 'quantity'] = weeks_sum_list.at[value, 0]
    print(df)
    #df['quantity'] = df.apply(lambda row: my_test(row[df.index.values], weeks_sum_list), axis=1)

    #Fill blank quantities with previous values (same week)
    #df.fillna(method='ffill')

    #for index, row in df.iterrows():
    #    if index in weeks_sum_list[0]
    #        indices = list(np.where(weeks_sum_list["date"] == date)[0])
    #        value = weeks_sum_list.iloc[indices].quantity
    #        index


dataframe = dl.load_sales_file('C:/Users/SM-Baerbar/Documents/GitHub/P5/CleanData/CleanedData.rpt')
print('--- data loaded ---')
print("--- %s seconds ---" % (time.time() - start_time))
SoerenSort(dataframe, '07Nov2017', 66)
print('--- result calculated ---')
print("--- %s seconds ---" % (time.time() - start_time))
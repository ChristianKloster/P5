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
import calendar
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

def month_change_distance_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    mcdist_list = []

    for index, row in df.iterrows():
        days_in_month = calendar.monthrange(row['date'].year, row['date'].month)[1]
        day = row['date'].day
        val = day if (days_in_month - day) > day else days_in_month - day
        mcdist_list.append(val)

    print(type(mcdist_list))

    return mcdist_list



def week_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns with date as index
    col_list = ["date", "quantity"]
    df = df[col_list]
    df.set_index('date', inplace=True)

    df = df.rolling('7d').sum()

    return df.quantity

def weekday_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    #Initialize a list for every weekday
    mon_list, tue_list, wed_list, thu_list, fri_list, sat_list, sun_list = ([] for i in range(7))


    # for every weekday in every row in df, set value to 1 if for weekday if it is this day, else 0
    for index, row in df.iterrows():
        mon_list.append(1) if row['date'].isoweekday() == 1 else mon_list.append(0)
        tue_list.append(1) if row['date'].isoweekday() == 2 else tue_list.append(0)
        wed_list.append(1) if row['date'].isoweekday() == 3 else wed_list.append(0)
        thu_list.append(1) if row['date'].isoweekday() == 4 else thu_list.append(0)
        fri_list.append(1) if row['date'].isoweekday() == 5 else fri_list.append(0)
        sat_list.append(1) if row['date'].isoweekday() == 6 else sat_list.append(0)
        sun_list.append(1) if row['date'].isoweekday() == 7 else sun_list.append(0)

    #Make a colum containing a true/false (1/0) value for each weekday (mon-sun)
    df['mon'] = mon_list
    df['tue'] = tue_list
    df['wed'] = wed_list
    df['thu'] = thu_list
    df['fri'] = fri_list
    df['sat'] = sat_list
    df['sun'] = sun_list

    #DF containing only true/false (1/0) value for each weekday (mon-sun)
    col_list = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    df = df[col_list]

    return df

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


dataframe = dl.load_sales_file('C:/Users/SMSpin/Documents/GitHub/P5/CleanData/CleanedData.rpt')
print('--- data loaded ---')
print("--- %s seconds ---" % (time.time() - start_time))
#SoerenSort(dataframe, '07Nov2017', 66)
#week_feature(dataframe,66)
#weekday_feature(dataframe, 66)
month_change_distance_feature(dataframe, 66)

print('--- result calculated ---')
print("--- %s seconds ---" % (time.time() - start_time))
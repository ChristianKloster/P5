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

#Returns a list of distances to the first date of the month (0 if first)
def month_first_dist_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    mfdist_list = []

    for index, row in df.iterrows():
        mfdist_list.append(row['date'].day - 1)

    print(mfdist_list)

    return mfdist_list

#Returns a list of distances to the last date of the month (0 if last)
def month_last_dist_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    mldist_list = []

    for index, row in df.iterrows():
        days_in_month = calendar.monthrange(row['date'].year, row['date'].month)[1]
        mldist_list.append((days_in_month - row['date'].day))

    print(mldist_list)

    return mldist_list

#Returns a list of distances to the first date of the month (0 if first)
def month_change_dist_feature(df, retailerID):
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

    return mcdist_list

#Returns a df column of summed quantities for the past 7 days
def week_quantity_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns with date as index
    col_list = ["date", "quantity"]
    df = df[col_list]
    df.set_index('date', inplace=True)

    df = df.rolling('7d').sum()

    return df.quantity

#Returns a df column of summed quantities for the past 30 days
def month_quantity_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns with date as index
    col_list = ["date", "quantity"]
    df = df[col_list]
    df.set_index('date', inplace=True)

    df = df.rolling('30d').sum()

    return df.quantity

#Returns a df containing 7 columns, one for each weekday
#If the date of the row is a someday, the value in the column representing someday will be 1, else 0
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

#Returns a df containing 12 columns, one for each month
#If the date of the row is in a somemonth, the value in the column representing somemonth will be 1, else 0
def month_feature(df, retailerID):
    df = df[df.retailerID == retailerID]

    # Df containing only required columns
    col_list = ['date']
    df = df[col_list]

    #Initialize a list for every month
    jan_list, feb_list, mar_list, apr_list, may_list, jun_list, jul_list, aug_list, sep_list, okt_list, nov_list, dec_list = ([] for i in range(12))


    # for every weekday in every row in df, set value to 1 if for weekday if it is this day, else 0
    for index, row in df.iterrows():
        jan_list.append(1) if row['date'].month == 1 else jan_list.append(0)
        feb_list.append(1) if row['date'].month == 2 else feb_list.append(0)
        mar_list.append(1) if row['date'].month == 3 else mar_list.append(0)
        apr_list.append(1) if row['date'].month == 4 else apr_list.append(0)
        may_list.append(1) if row['date'].month == 5 else may_list.append(0)
        jun_list.append(1) if row['date'].month == 6 else jun_list.append(0)
        jul_list.append(1) if row['date'].month == 7 else jul_list.append(0)
        aug_list.append(1) if row['date'].month == 8 else aug_list.append(0)
        sep_list.append(1) if row['date'].month == 9 else sep_list.append(0)
        okt_list.append(1) if row['date'].month == 10 else okt_list.append(0)
        nov_list.append(1) if row['date'].month == 11 else nov_list.append(0)
        dec_list.append(1) if row['date'].month == 12 else dec_list.append(0)

    #Make a colum containing a true/false (1/0) value for each weekday (mon-sun)
    df['jan'] = jan_list
    df['feb'] = feb_list
    df['mar'] = mar_list
    df['apr'] = apr_list
    df['may'] = may_list
    df['jun'] = jun_list
    df['jul'] = jul_list
    df['aug'] = aug_list
    df['sep'] = sep_list
    df['okt'] = okt_list
    df['nov'] = nov_list
    df['dec'] = dec_list

    #DF containing only true/false (1/0) value for each weekday (mon-sun)
    col_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
    df = df[col_list]

    return df

#Load clean data
dataframe = dl.load_sales_file('C:/Users/SMSpin/Documents/GitHub/P5/CleanData/CleanedData.rpt')
print('--- data loaded ---')
print("--- %s seconds ---" % (time.time() - start_time))
#SoerenSort(dataframe, '07Nov2017', 66)
#week_quantity_feature(dataframe,66)
#month_change_dist_feature(dataframe, 66)
#month_quantity_feature(dataframe, 66)
#weekday_feature(dataframe, 66)
#month_feature(dataframe, 66)
#month_change_dist_feature(dataframe, 66)
month_first_dist_feature(dataframe, 66)
month_last_dist_feature(dataframe, 66)

print('--- result calculated ---')
print("--- %s seconds ---" % (time.time() - start_time))
import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from collections import OrderedDict
from retailerdata import retailerData as rd
import os
import sys

def between_days_sum(df, start_date, end_date):
    d = df[(df.date.dt.day >= start_date) & (df.date.dt.day <=end_date)]
    return d.turnover.sum()

#Metode der modtager laver et diagram som viser fordelingen af salget over farver og størrelser hos en enkelt vare.
#Metoden modtager et input data og et navn. Dataen skal også indeholde "row keys" som bliver farven på søjlen.
def histogram(inputdata, titl, xlabel, ylabel):
    df = pd.DataFrame(inputdata, index=list(range(1,32)))
    colors = ['blue']
    x = df.plot(kind='bar', color=colors, edgecolor='black', title=titl)
    x.set_xlabel(xlabel)
    x.set_ylabel(ylabel)
    x.legend([xlabel])
    plt.tight_layout()

def plot_per_day(dataframe, unit, year, month_is_specific = False):
    df = dataframe
    raw_data = []

    if unit == 'Quantity':
        for day in range(1, 32):
            d = df[df.date.dt.day == day]
            raw_data.append(d.quantity.sum())
    elif unit == 'Turnover':
        for day in range(1, 32):
            d = df[df.date.dt.day == day]
            raw_data.append(d.turnover.sum())
    else:
        sys.exit("Incorrect unit")

    month_str = df.iloc[0].date.strftime("%B")
    print(unit + ' ' + month_str + ' ' + str(year))
    print(raw_data)
    title = month_str + ' ' + str(year) if month_is_specific else 'All months'
    histogram(raw_data, title, 'Date', unit)

    if month_is_specific:
        file_name = 'PLC/month_sale/' + unit + '/' + unit + '_' + str(year) + '_' + str(
            df.iloc[0].date.month) + '(' + month_str + ')'
    else:
        file_name = 'PLC/month_sale/' + unit + '/' + unit + '_all_available_months'
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(file_name)
    print('Saved plot' + '\n----------------------------')

def plot_per_day_individual_months(dataframe, unit, year):
    first_available_month = 6 if (year == 2016) else 1
    last_available_month = 12 if (year == 2016) else 9

    for month in range(first_available_month, last_available_month + 1):
        df = dataframe[(dataframe.date.dt.month == month) & (dataframe.date.dt.year == year)]
        plot_per_day(df, unit, year, True)


dataframe = dl.load_sales_file('C:/Users/SM-Baerbar/Documents/GitHub/P5/CleanData/CleanedData.rpt')
plot_per_day(dataframe, 'Quantity', 2016)
plot_per_day(dataframe, 'Turnover', 2016)
plot_per_day_individual_months(dataframe, 'Quantity', 2016)
plot_per_day_individual_months(dataframe, 'Quantity', 2017)
plot_per_day_individual_months(dataframe, 'Turnover', 2016)
plot_per_day_individual_months(dataframe, 'Turnover', 2017)
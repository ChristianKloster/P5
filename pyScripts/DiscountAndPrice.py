import pandas as pd
import numpy as np
import dataloader as dl

df = dl.load_sales_file(r'C:\Git repo\P5\CleanData\CleanedData.rpt')

def get_avg(lst):
    if len(lst)>0:
        return sum(lst) / len(lst)
    else:
        return 0

def discount_to_percent(dataframe):
    percent_list = []

    for i in range(0, len(dataframe.discount)):
        if dataframe.discount[i] == 0:
            percent_list.append(0)

        else:
            percent_list.append(-1 * dataframe.dicount/(-1*dataframe.discount[i]+dataframe.turnover[i])*100)
            print(-1 * dataframe.dicount/(-1 * dataframe.discount[i]+dataframe.turnover[i])*100)

    return percent_list


def get_avg_price_in_style(dataframe):
    styles = df.styleNumber.unique()
    all_styles_avg_price = []
    avg_price_for_trans = []

    for style in styles: #Finding the average price
        indexes = df.index[df.styleNumber == style].tolist()
        single_style_list= []

        for index in indexes:
            value=dataframe.turnover[index]+(dataframe.discount[index]*-1)
            single_style_list.append(value)

        all_styles_avg_price.append(get_avg(single_style_list))

    for i in range(0, len(dataframe)-1): #searches the all_styles_avg_price list for the price at the index of the style in the styles list and inserts that on the spot in the main feature file where set style occurs .
        avg_value =all_styles_avg_price[styles.index[dataframe.styleNumber[i]]]
        avg_price_for_trans.append(avg_value)


print(get_avg_price_in_style(df))


def create_avg_list(dataframe, parameter):
    full_list = []
    single_day_list = []

    for i in range(0, len(dataframe.date)):
        day = dataframe.date[i]

        if i == len(dataframe.date) or dataframe.date[i+1] != day:
            full_list.append(get_avg(single_day_list))
            print(len(full_list))
            single_day_list = []
        else:
            for i in dataframe.quantity[i]: #Adds the item one time for each quantity.
                single_day_list.append(dataframe[parameter][i])
    return full_list


#create_avg_list(df, 'discount')


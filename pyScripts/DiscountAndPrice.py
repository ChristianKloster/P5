import pandas as pd
import numpy as np
import dataloader as dl

df = dl.load_sales_file(r'C:\Git repo\P5\CleanData\CleanedData.rpt')

output_list = []


def get_avg(lst):
    return sum(lst) / len(lst)


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
            single_day_list.append(dataframe[parameter][i])
    return full_list


create_avg_list(df, 'discount')

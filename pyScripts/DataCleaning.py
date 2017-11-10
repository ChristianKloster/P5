import pandas as pd
import numpy as np
import dataloader as dl
from retailerdata import retailerData as rd

#Open the same the clean file to make changes
df = dl.load_sales_file(r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\CleanedData_New.rpt')
rdf = rd(r'C:\Users\Christian\Desktop\Min Git mappe\P5\retailers_w_coords.rpt')

#For fresh start
#df = dl.load_sales_files(dl.get_all_files(r'C:\Git repo\P5\GOFACT_DATA\Sales_20'))

#This method opens the meta file and notes which data has been removed.
def meta_writer(parameter, word, path):
    with open(path + r'meta.txt', "a") as myfile:
        myfile.write(parameter +': '+ str(word) + '\n')

#df: unclean data. parameter: Columntype. words: Entries you want to filter out. path: The path to the cleandata folder in your git folder.
def data_cleaner(df, parameter, words, path): #Can handle 0, one or more words. 0 Returns the same dataframe.
    if len(words) > 1:
        for word in words:
            df = df[df[parameter] != word]#Keeps rows where paramter isn't word.
            meta_writer(parameter, word, path)
    elif len(words) == 1:
        df = df[df[parameter] != words[0]]
        meta_writer(parameter, words[0], path)
    else:
        print('Didn\'t do nothing')

    df.to_csv(path_or_buf=path+'CleanedData.rpt', index = False, sep=';',encoding='utf-8')


def add_chainId(df,rdf):
    chainId_list = []

    for i in range(0,len(df)):
        chainId_list.append(rdf.get_chainnumber(df.retailerID[i]))

    df['chainID'] = chainId_list
    return df

add_chainId(df, rdf).to_csv(path_or_buf=r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\CleanedData_New.rpt', index = False, sep=';',encoding='utf-8')

#ALREADY CALLED:
#rddf = rd(r'C:\Git repo\P5\Retailers_w_coords.rpt')
#retailerID = rddf.get_retailers_from_country('Sweden')
#meta_writer('Sweden','',r'C:\Git repo\P5\CleanData\\')
#data_cleaner(df, 'retailerID', retailerID['id'], r'C:\Git repo\P5\CleanData\\')
#retailerID = rddf.get_retailers_from_country('Holland')
#meta_writer('Holland', '',r'C:\Git repo\P5\CleanData\\')
#data_cleaner(df, 'retailerID', retailerID['id'], r'C:\Git repo\P5\CleanData\\')
#retailerID = rddf.get_retailers_from_country('Norway')
#meta_writer('Norway', '',r'C:\Git repo\P5\CleanData\\')
#data_cleaner(df, 'retailerID', retailerID['id'], r'C:\Git repo\P5\CleanData\\')
#retailerID = rddf.get_retailers_from_country('Finland')
#meta_writer('Finland', '',r'C:\Git repo\P5\CleanData\\')
#data_cleaner(df, 'retailerID', retailerID['id'], r'C:\Git repo\P5\CleanData\\')
#retailerID = rddf.get_retailers_from_country('Germany')
#meta_writer('Germany', '',r'C:\Git repo\P5\CleanData\\')
#data_cleaner(df, 'retailerID', retailerID['id'], r'C:\Git repo\P5\CleanData\\')

#data_cleaner(df, 'SupplierItemgroupName', ['Dummy','Unknown'], r'C:\Git repo\P5\CleanData\\')

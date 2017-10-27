import pandas as pd
import numpy as np
import dataloader as dl
from retailerdata import retailerData as rd

#Open the same the clean file to make changes
df = dl.load_sales_file(r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\CleanedData.rpt')

#For fresh start
#df = dl.load_sales_files(dl.get_all_files(r'C:\Users\Christian\Desktop\Min Git mappe\P5\GOFACT_DATA\Sales_20'))

#This method opens the meta file and notes which data has been removed.
def meta_writer(parameter, word, path):
    with open(path + 'meta.txt', "a") as myfile:
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

DisList = df['quantity'].tolist()
DisList.sort()
print(DisList[len(DisList)-5])


#ALREADY CALLED:
#data_cleaner(df, 'quantity', [400,-400], r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\')

#rddf = rd(r'C:\Users\Christian\Desktop\Min Git mappe\P5\Retailers_w_coords.rpt')
#retailerID = rddf.get_retailers_from_country('Denmark')
#data_cleaner(df, 'retailerID', retailerID['id'], r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\')

#data_cleaner(df, 'SupplierItemgroupName', ['Dummy','Unknown'], r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\')

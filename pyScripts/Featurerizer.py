import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from calendar import monthrange
import time
import numpy as np
import calendar


#Returns a dataframe column of distances to the first date of the month (0 if first)
def month_first_dist_feature(df):
    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    mfdist_list = []

    for index, row in df.iterrows():
        mfdist_list.append(row['date'].day - 1)

    df['mfdist_list'] = mfdist_list

    return df.mfdist_list

#Returns a list of distances to the first date of the next month (0 if last)
def month_next_first_dist_feature(df):
    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    mldist_list = []

    for index, row in df.iterrows():
        days_in_month = calendar.monthrange(row['date'].year, row['date'].month)[1]
        mldist_list.append((days_in_month + 1 - row['date'].day))

    df['mldist_list'] = mldist_list

    return df.mldist_list

#Returns a list of distances to the nearest month change
def month_change_dist_feature(df):
    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    mcdist_list = []

    for index, row in df.iterrows():
        days_in_month = calendar.monthrange(row['date'].year, row['date'].month)[1]
        day = row['date'].day
        val = day - 1 if (days_in_month - day) > day else days_in_month + 1 - day
        mcdist_list.append(val)

    df['mcdist_list'] = mcdist_list

    return df.mcdist_list

#Returns a df column of summed quantities for the past 7 days
def week_quantity_feature(df):
    # Df containing only required columns with date as index
    col_list = ["date", "quantity"]
    df = df[col_list]
    df.set_index('date', inplace=True)

    df = df.rolling('7d').sum()

    return df.quantity

#Returns a df column of summed quantities for the past 30 days
def month_quantity_feature(df):
    # Df containing only required columns with date as index
    col_list = ["date", "quantity"]
    df = df[col_list]
    df.set_index('date', inplace=True)

    df = df.rolling('30d').sum()

    return df.quantity

#Returns a df containing 7 columns, one for each weekday
#If the date of the row is a someday, the value in the column representing someday will be 1, else 0
def weekday_feature(df):
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
def month_feature(df):
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

#Returns a list all discounts as percent
def discount_to_percent(dataframe):
    df = dataframe[['discount','turnover']]

    df['discountP'] = abs(df['discount'])/(abs(df['discount'])+abs(df['turnover']))*100

    return df['discountP']

#Returns a list of avg style price for each transaction
def get_avg_price_in_style(dataframe):
    df = dataframe[['styleNumber','turnover','discount','quantity']]
    df = df.groupby('styleNumber').sum(numeric_only = True)
    df['avg_style_price'] = (abs(df['turnover'])+abs(df['discount']))/abs(df['quantity'])

    return df['avg_style_price']

#Returns a list with average day price for each transaction.
def create_avg_list(dataframe):
    df = dataframe[['date','turnover','discount','quantity']]
    df = df.groupby('date').sum(numeric_only = True)
    df['avg_day_price'] = (abs(df['turnover'])+abs(df['discount']))/abs(df['quantity'])

    return df['avg_day_price']

def featureplcBD(df, id):
    #Relativt store udslag i hældningen, ikke nær så præcis som CD metoden, kan produceres for dags dato
    if id in df.productID:
        df = df[df.productID == id]
        df = df[df.styleNumber == df['styleNumber'].iloc[0]]
    else:
        df = df[df.styleNumber == id]
    df = df[df.quantity >= 0]
    df = df.groupby('date', as_index=False).sum()
    prev = df.shift(periods = 1)
    test = df['date'] - prev['date']
    timedif = pd.DataFrame(columns=['date'], index=test.index)
    test = test.fillna(value=1)
    for x in test.index:
        timedif.iloc[x] = int(test.iloc[x].days)
    timedif.iloc[0] = 1
    tempslope = (df['quantity'] - prev['quantity'])/timedif['date']
    slope = tempslope.reindex(df['date'])
    for x in tempslope.index:
        slope.iloc[x] = tempslope.iloc[x]
    slope = slope.fillna(value=0)
    slope = pd.DataFrame(slope, columns=['slopeBD'])
    return slope

def featureplcCD(df, id):
    #Denne giver mindre udslag i hældningerne og burde give et mere korrekt udtryk, men den kan ikke produceres for dags dato
    if id in df.productID:
        df = df[df.productID == id]
        df = df[df.styleNumber == df['styleNumber'].iloc[0]]
    else:
        df = df[df.styleNumber == id]
    df = df[df.quantity >= 0]
    df = df.groupby('date', as_index=False).sum()
    prev = df.shift(periods = 1)
    next = df.shift(periods=-1)
    test = next['date'] - prev['date']
    timedif = pd.DataFrame(columns=['date'], index=test.index)
    test = test.fillna(value=1)
    for x in test.index:
        timedif.iloc[x] = int(test.iloc[x].days)
    timedif.iloc[0] = 1
    timedif.iloc[-1] = 1
    tempslope = (next['quantity'] - prev['quantity'])/timedif['date']
    slope = tempslope.reindex(df['date'])
    for x in tempslope.index:
        slope.iloc[x] = tempslope.iloc[x]
    slope = slope.fillna(value=0)
    slope = pd.DataFrame(slope, columns=['slopeCD'])
    return slope

def featurealder(df, id):
    if id in df.productID:
        df = df[df.productID == id]
    else:
        df = df[df.styleNumber == id]
    df = df[df.quantity >= 0]
    firstdate = df['date'].iloc[0]
    lifetimetemp = pd.DataFrame(columns=['lifetime'], index=df.index)
    for x in lifetimetemp.index:
        lifetimetemp.loc[x] = int((df['date'].loc[x] - firstdate).days)
    lifetime = lifetimetemp.reindex(df['date'])
    for x in range(0, lifetimetemp.shape[0]):
        lifetime.iloc[x] = lifetimetemp.iloc[x]
    today = pd.DataFrame([int((pd.to_datetime('today') - firstdate).days)], index=[pd.to_datetime('today')], columns=['lifetime'])
    lifetime = pd.concat([lifetime, today])
    return(lifetime)

def featureacceleration(df, id):
    if id in df.productID:
        df = df[df.productID == id]
    else:
        df = df[df.styleNumber == id]
    df = df[df.quantity >= 0]
    df = df.groupby('date', as_index=False).sum()
    prev = df.shift(periods=1)
    test = df['date'] - prev['date']
    timedif = pd.DataFrame(columns=['date'], index=test.index)
    test = test.fillna(value=1)
    for x in test.index:
        timedif.iloc[x] = int(test.iloc[x].days)
    timedif.iloc[0] = 1
    slope = (df['quantity'] - prev['quantity']) / timedif['date']
    slope = slope.fillna(value=0)
    slope = pd.DataFrame(slope, columns=['slope'])
    prev = slope.shift(periods=1)
    tempaccel = (slope['slope'] - prev['slope'])/timedif['date']
    acceleration = tempaccel.reindex(slope.index)
    for x in tempaccel.index:
        acceleration.iloc[x] = tempaccel.iloc[x]
    acceleration = acceleration.fillna(value=0)
    return acceleration



col_name = 'color_popularity'

class ColorFeatureProvider:
    def __init__(self, df, window = '7d'):
        df = df.copy()
        df.colorname = df.colorname.str.lower().str.strip()
        df_table = pd.pivot_table(df, values='quantity', index=['date'], columns=['colorname'], aggfunc=np.sum).fillna(0)
        df_table = df_table.rolling(window).mean()
        self.table = df_table

    def get_color_popularity(self, color=None, date=None):
        if color == None and date == None:
            return self.table
        elif color == None:
            return self.table.loc[date,:]
        elif date == None:
            return self.table.loc[:,color]
        else:
            return self.table.loc[date, color]

def make_feature_col(df, window = '7d'):
    cfp = ColorFeatureProvider(df, window)
    data = df.copy()
    data.colorname = data.colorname.str.lower().str.strip()
    data = data.dropna(axis=0, how='any')

    data[col_name] = np.nan

    data[col_name] = tuple(map(lambda color, date: cfp.get_color_popularity(color = color, date = date), data['colorname'], data['date']))

    return data

def get_feature_name():
    return col_name


class SizeFeature:
    def __init__(self):

# disse bør nok laves til nogle tabeller vi kan præsentere:

        self.mapping_ch1 = {'XS' : 1,
                            'S' : 2,
                            'S/M' : 3,
                            'M' : 4,
                            'M/L' : 5,
                            'L' : 6,
                            'L/XL' : 7,
                            'XL' : 8,
                            'XXL' : 10,
                            '40' : 1 ,
                            '42' : 2,
                            '44' : 3,
                            '46' : 4,
                            '48' : 5,
                            '50' : 6,
                            '52' : 7,
                            '54' : 8,
                            '56' : 9,
                            '105' : 2,
                            '115' : 4,
                            '125' : 6,
                            'ONE SIZE' : 0,
                            'Unknown' : 9999}

        self.mapping_ch2_w = {'XXS' : 1,
                            'XS' : 2,
                            'XS/S' : 3, 
                            'S' : 4,
                            'S/M' : 5,
                            'M' : 6,
                            'M/L' : 7,
                            'L' : 8,
                            'L/XL' : 9,
                            'XL' : 10,
                            'XXL' : 11,
                            '24\"' : 1,
                            '25\"' : 2,
                            '26\"' : 3,
                            '27\"' : 4,
                            '28\"' : 5,
                            '29\"' : 6,
                            '30\"' : 8,
                            '31\"' : 9,
                            '32\"' : 10,
                            '33\"' : 11,
                            '34\"' : 12,
                            '36\"' : 13,
                            '37-40' : 7,
                            '41-46' : 9,
                            '36' : 6,
                            '37' : 6,
                            '38' : 6,
                            '39' : 7,
                            '40' : 8,
                            '41' : 8,
                            '42' : 8,
                            '43' : 9,
                            '44' : 10,
                            '45' : 10,
                            '46' : 10,
                            '48' : 11,
                            '50' : 12,
                            '52' : 12,
                            '54' : 13,
                            '56' : 13,
                            '85' : 1,
                            '90' : 2,
                            '95' : 4,
                            '100' : 6,
                            '105' : 9,
                            '*** ONE SIZE ***' : 0,
                            'ONE SIZE' : 0,
                            'ONE' : 0,
                            'Unknown' : 9999 }

        self.mapping_ch2_m = {'XXS' : 1,
                            'XS' : 2,
                            'XS/S' : 3, 
                            'S' : 4,
                            'S/M' : 5,
                            'M' : 6,
                            'M/L' : 7,
                            'L' : 8,
                            'L/XL' : 9,
                            'XL' : 10,
                            'XXL' : 11,
                            '24\"' : 1,
                            '25\"' : 2,
                            '26\"' : 3,
                            '27\"' : 4,
                            '28\"' : 6,
                            '29\"' : 8,
                            '30\"' : 9,
                            '31\"' : 10,
                            '32\"' : 11,
                            '33\"' : 11,
                            '34\"' : 12,
                            '36\"' : 13,
                            '37-40' : 1,
                            '41-46' : 2,
                            '36' : 1,
                            '37' : 1,
                            '38' : 1,
                            '39' : 1,
                            '40' : 1,
                            '41' : 1,
                            '42' : 1,
                            '43' : 1,
                            '44' : 2,
                            '45' : 3,
                            '46' : 4,
                            '48' : 6,
                            '50' : 8,
                            '52' : 10,
                            '54' : 11,
                            '56' : 12,
                            '85' : 1,
                            '90' : 4,
                            '95' : 8,
                            '100' : 10,
                            '105' : 11,
                            '*** ONE SIZE ***' : 0,
                            'ONE SIZE' : 0,
                            'ONE' : 0,
                            'Unknown' : 9999}

        self.mapping_ch3  = {'XS' : 1,
                            'XS/S': 2,
                            'S' : 3,
                            'S/M' : 4,   
                            'M' : 5,
                            'M/L' :6,  
                            'L' :7,
                            'ONE SIZE' : 0,
                            'Unknown' : 9999}

# zizzi
# Vores styles er altid udviklet efter europæiske måle standarder.
# https://www.zizzi.dk/hjaelp/guides/stoerrelsesguide

# samsoe women : http://www.sizeguide.net/womens-clothing-sizes-international-conversion-chart.html
# http://www.samsoe.com/da/support/size-guide/mens-jeans.html
 

    def get_size_feature(self, size, chainid, male = False):
        if chainid == 1:
            return self.mapping_ch1[size]
        elif chainid == 2:
            if male:
                return self.mapping_ch2_m[size]
            else:
                return self.mapping_ch2_w[size]
        elif chainid == 3:
            return self.mapping_ch3[size]
        else: 
            return 8888 

# requires that chainid and ismale is present
def make_sizefeature_col(df):
    sf = SizeFeature()

    data = df.copy()
    data['size_scale'] = tuple(map(lambda size, chainid, ismale: sf.get_size_feature(size=size, chainid = chainid, male = ismale), data['size'], data['chainid'], date['ismale']))
    return data


mens_itemgroup = ['MEN - ACCESSORIES',
                'MEN - BASIC - DON\'T USE',
                'MEN - BELTS/SCARF/TIE',
                'MEN - BLAZER',
                'MEN - CARDIGAN',
                'MEN - JACKETS',
                'MEN - JEANS',
                'MEN - KNIT',
                'MEN - PANTS',
                'MEN - POLO',
                'MEN - SHIRTS',
                'MEN - SHOES', 
                'MEN - SHORTS',
                'MEN - SWEAT',               
                'MEN - T-SHIRTS',            
                'MEN - UNDERWEAR/SOCKS']    

def is_male(itemgroupname):
    return 1 if itemgroupname in mens_itemgroup else 0

def make_ismale_col(df):
    data = df.copy()
    data['ismale'] = tuple(map(lambda itemgroup: is_male(itemgroup), data['SupplierItemgroupName']))
    return data

def make_isfemale_col(df):
    data = df.copy()
    data['isfemale'] = tuple(map(lambda itemgroup: 1 - is_male(itemgroup), data['SupplierItemgroupName']))
    return data

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#Load clean data
#sm_dir = 'C:/Users/SMSpin/Documents/GitHub/P5/CleanData/CleanedData.rpt'
kloster_dir = r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\CleanedData_New.rpt'

dataframe = dl.load_sales_file(kloster_dir)

print(get_avg_price_in_style(dataframe))

print(create_avg_list(dataframe))

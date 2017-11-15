import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from calendar import monthrange
import time
import numpy as np
import calendar

def today_turnover(df):
    todayturnover = pd.DataFrame(columns=['turnoverChain'], index=df.index)
    todayturnover[['date', 'chainID', 'turnover']] = df[['date', 'chainID', 'turnover']]
    chains = todayturnover['chainID'].unique()
    datedata = pd.DataFrame(columns=[['date', 'chainID', 'quantityChain']])

    for chain in chains:
        chaindf = todayturnover[todayturnover.chainID == chain]
        chaindf = chaindf.groupby('date', as_index=False).sum(numeric_only = True)
        chaindf['turrnoverChain'] = chaindf['turnover']
        chaindf['chainID'] = chain
        chaindf = chaindf[['date', 'chainID', 'turnoverChain']]
        datedata = datedata.append(chaindf)
        todayturnover.merge(datedata)
    return todayturnover['turnoverChain']

def today_quantity(df):
    todayquantity = pd.DataFrame(columns=['quantityChain'], index=df.index)
    todayquantity[['date', 'chainID', 'quantity', 'productID']] = df[['date', 'chainID', 'quantity', 'productID']]
    chains = todayquantity['chainID'].unique()
    datedata = pd.DataFrame(columns=[['date', 'chainID', 'productID', 'quantityChain']])

    for chain in chains:
        chaindf = todayquantity[todayquantity.chainID == chain]
        products = chaindf['productID'].unique()
        for product in products:
            productdf = chaindf[chaindf.productID == product]
            productdf = productdf.groupby('date', as_index=False).sum()
            productdf['quantityChain'] = productdf['quantity']
            productdf['productID'] = product
            productdf['chainID'] = chain
            productdf = productdf[['chainID', 'date' ,'productID', 'quantityChain']]
            datedata = datedata.append(productdf)
    todayquantity.merge(datedata)
    return todayquantity['quantityChain']

def quantity_period(df, number_periods = 3, length_period = 4):
    col_list = ["date", "quantity", "retailerID", "productID"]
    period_col = []
    for i in range (0, number_periods):
        period_col.append('quantityPeriod{0}'.format(i+1))
    quantityPerioddf = pd.DataFrame(columns=period_col, index=df.index)
    quantityPerioddf[col_list] = df[col_list]
    retailers = quantityPerioddf["retailerID"].unique()

    for retailer in retailers:
        retailerdf = quantityPerioddf[quantityPerioddf.retailerID == retailer]
        products = retailerdf["productID"].unique()

        for product in products:
            productdf = retailerdf[retailerdf.productID == product]
            for i in range(1, number_periods+1):
                productsumdf = productdf.rolling('{0}d'.format(length_period*i), on='date').sum()
                if i>1:
                    productdf['quantityPeriod{0}'.format(i)] = productsumdf['quantity']-productdf['quantityPeriod{0}'.format(i-1)]
                else:
                    productdf['quantityPeriod{0}'.format(i)] = productsumdf['quantity']
                quantityPerioddf.update(productdf)

    return quantityPerioddf[period_col]

def stigning_period(df, number_periods = 3, length_period = 4):
    col_list = ["date", "quantity", "retailerID", "productID"]
    period_col = []
    for i in range (1, number_periods+1):
        period_col.append('quantityPeriod{0}'.format(i))
        if i > 1:
            period_col.append('P{0}-P{1}'.format(i - 1, i))
    quantityPerioddf = pd.DataFrame(columns=period_col, index=df.index)
    quantityPerioddf[col_list] = df[col_list]
    retailers = quantityPerioddf["retailerID"].unique()

    for retailer in retailers:
        retailerdf = quantityPerioddf[quantityPerioddf.retailerID == retailer]
        products = retailerdf["productID"].unique()

        for product in products:
            productdf = retailerdf[retailerdf.productID == product]
            for i in range(1, number_periods+1):
                productsumdf = productdf.rolling('{0}d'.format(length_period*i), on='date').sum()
                productdf['quantityPeriod{0}'.format(i)] = productsumdf['quantity']
                if i>1:
                    productdf['quantityPeriod{0}'.format(i)] = productsumdf['quantity'] - productdf['quantityPeriod{0}'.format(i - 1)]
                    productdf['P{0}-P{1}'.format(i-1, i)] = \
                        (productdf['quantityPeriod{0}'.format(i-1)] - productdf['quantityPeriod{0}'.format(i)])/length_period
                else:
                    productdf['quantityPeriod{0}'.format(i)] = productsumdf['quantity']
                quantityPerioddf.update(productdf)

    return quantityPerioddf[period_col]

#Returns a dataframe column of distances to the first date of the month (0 if first)
def month_first_dist_feature(df):
    # Df containing only required columns with date as index
    col_list = ['date']
    df = df[col_list]

    mfdist_list = []

    for index, row in df.iterrows():
        mfdist_list.append(row['date'].day - 1)

    df['mfdist_list'] = mfdist_list

    return df['mfdist_list']

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

    return df['mldist_list']

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
    test = df['mcdist_list']
    return df['mcdist_list']

#Returns a df column of summed quantities for the past 7 days
def week_quantity_feature(df):
    # Df containing only required columns with date as index
    col_list = ["date", "quantity", "retailerID", "productID"]
    last7df = pd.DataFrame(columns=['quantityLast7'], index=df.index)
    last7df[col_list] = df[col_list]
    retailers = last7df["retailerID"].unique()

    for retailer in retailers:
        retailerdf = last7df[last7df.retailerID == retailer]
        products = retailerdf["productID"].unique()
        for product in products:
            productdf = retailerdf[retailerdf.productID == product]
            productdf = productdf.rolling('7d', on='date').sum()
            productdf['quantityLast7'] = productdf['quantity']
            last7df.update(productdf)

    return last7df['quantityLast7']

#Returns a df column of summed quantities for the past 30 days
def month_quantity_feature(df):
    # Df containing only required columns with date as index
    col_list = ["date", "quantity", "retailerID", "productID"]
    last30df = pd.DataFrame(columns=['quantityLast30'], index=df.index)
    last30df[col_list] = df[col_list]
    retailers = last30df["retailerID"].unique()

    for retailer in retailers:
        retailerdf = last30df[last30df.retailerID == retailer]
        products = retailerdf["productID"].unique()
        for product in products:
            productdf = retailerdf[retailerdf.productID == product]
            productdf = productdf.rolling('30d', on='date').sum()
            productdf['quantityLast30'] = productdf['quantity']
            last30df.update(productdf)

    return last30df['quantityLast30']

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

    #DF containing only true/false (1/0) value for each month (jan-dec)
    col_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
    df = df[col_list]

    return df

#Returns a list all discounts as percent
def discount_to_percent(dataframe):
    df = dataframe[['discount','turnover']]
    discount = pd.DataFrame(columns=['discountP'], index=df.index)
    discount['discountP'] = abs(df['discount'])/(abs(df['discount'])+abs(df['turnover']))*100

    return discount['discountP']

#Returns a list of avg style price for each transaction
def get_avg_price_in_style(dataframe):
    df = dataframe[['styleNumber','turnover','discount','quantity']]
    avgprice = pd.DataFrame(columns=['avg_style_price'], index=df.index)

    avgprice['avg_style_price'] = (abs(df['turnover'])+abs(df['discount']))/abs(df['quantity'])

    return avgprice['avg_style_price']

#Returns a list with average day price for each transaction.
#Executive grazt
def create_avg_list(dataframe):
    df = dataframe[['date','turnover','discount','quantity']]
    df = df.groupby('date', as_index= False).sum(numeric_only = True)
    df['avg_day_price'] = (abs(df['turnover'])+abs(df['discount']))/abs(df['quantity'])
    avg_lst= []
    x=0

    for i in dataframe.index:
        if dataframe.date[i] == df.date[x]:
            avg_lst.append(df.avg_day_price[x])
        elif x<df.shape[0]-1:
            x+=1

    print(len(avg_lst))
    return avg_lst

def featureplcBD(df):
    #Relativt store udslag i hældningen, ikke nær så præcis som CD metoden, kan produceres for dags dato
    df = df[df.quantity >= 0]
    df = df[['date', 'chainID', 'quantity', 'styleNumber']]
    chains = df['chainID'].unique()
    slope = pd.DataFrame(columns=['slopeBD'], index=df.index)
    slope[['date','styleNumber','chainID']] = df[['date','styleNumber','chainID']]

    for chain in chains:
        chaindf = df[df.chainID == chain]
        styles = chaindf['styleNumber'].unique()
        chaindata = slope[slope.chainID == chain]
        for style in styles:
            styledf = chaindf[chaindf.styleNumber == style]
            styledf = styledf.groupby('date', as_index=False).sum(numeric_only = True)
            styledf['chainID'] = chain
            prev = styledf[['date', 'quantity']].shift(periods = 1)
            timedifference = (styledf['date'] - prev['date']).dt.days
            timedifference = timedifference.fillna(value=1)
            timedifference.iloc[0] = 1
            styleslope = (styledf['quantity'] - prev['quantity'])/timedifference
            styledata = chaindata[chaindata.styleNumber == style]
            for x in styleslope.index:
                datedata = styledata[styledata.date == styledf['date'].iloc[x]]
                datedata['slopeBD'] = styleslope.iloc[x]
            slope.update(datedata)
    slope['slopeBD'] = slope['slopeBD'].fillna(value=0)
    return slope['slopeBD']

def featureplcCD(df):
    #Denne giver mindre udslag i hældningerne og burde give et mere korrekt udtryk, men den kan ikke produceres for dags dato
    df = df[df.quantity >= 0]
    df = df[['date', 'chainID', 'quantity', 'styleNumber']]
    chains = df['chainID'].unique()
    slope = pd.DataFrame(columns=['slopeCD'], index=df.index)
    slope[['date','styleNumber','chainID']] = df[['date','styleNumber','chainID']]

    for chain in chains:
        chaindf = df[df.chainID == chain]
        styles = chaindf['styleNumber'].unique()
        chaindata = slope[slope.chainID == chain]
        for style in styles:
            styledf = chaindf[chaindf.styleNumber == style]
            styledf = styledf.groupby('date', as_index=False).sum(numeric_only = True)
            styledf['chainID'] = chain
            prev = styledf[['date', 'quantity']].shift(periods = 1)
            next = styledf[['date', 'quantity']].shift(periods = -1)
            timedifference = (next['date'] - prev['date']).dt.days
            timedifference = timedifference.fillna(value=1)
            timedifference.iloc[0] = 1
            styleslope = (next['quantity'] - prev['quantity'])/timedifference
            styledata = chaindata[chaindata.styleNumber == style]
            for x in styleslope.index:
                midlertidigdato = styledata[styledata.date == styledf['date'].iloc[x]]
                midlertidigdato['slopeCD'] = styleslope.iloc[x]
            slope.update(midlertidigdato)
    slope['slopeCD'] = slope['slopeCD'].fillna(value=0)
    return slope['slopeCD']

# def featurealder(df):
#     df = df[df.quantity >= 0]
#     lifetimes = pd.DataFrame(columns=[['lifetimeChain', 'lifetimeRetailer']], index=df.index)
#     lifetimes[['styleNumber', 'date', 'chainID', 'retailerID']] = df[['styleNumber', 'date', 'chainID', 'retailerID']]
#     chains = df['chainID'].unique()

#     for chain in chains:
#         chaindf = df[df.chainID == chain]
#         styles = chaindf['styleNumber'].unique()
#         chaindata = lifetimes[lifetimes.chainID == chain]
#         for style in styles:
#             styledata = chaindata[chaindata.styleNumber == style]
#             firstdate = styledata['date'].iloc[0]
#             for x in styledata.index:
#                 lifetimes['lifetimeChain'].loc[x] = int((styledata['date'].loc[x] - firstdate).days)

#     retailers = df['retailerID'].unique()
#     for retailer in retailers:
#         retailerdf = df[df.retailerID == retailer]
#         styles = retailerdf['styleNumber'].unique()
#         for style in styles:
#             styledata = retailerdf[retailerdf.styleNumber == style]
#             firstdate = styledata['date'].iloc[0]
#             for x in styledata.index:
#                 lifetimes['lifetimeRetailer'].loc[x] = int((styledata['date'].loc[x] - firstdate).days)

#     return lifetimes[['lifetimeChain', 'lifetimeRetailer']]

# laver alder på stylenumber baseret på df
def feature_age(df):
    data = df[['date', 'styleNumber']].copy()

    firstdates = data.groupby('styleNumber').first()
    print(firstdates)

    data['first'] = tuple(map(lambda style: firstdates.loc[style,'date'], data['styleNumber']))
    data['styleage'] = (data['date'] - data['first']).dt.days

    return data['styleage']



def featureacceleration(df):
    df = df[df.quantity >= 0]
    df = df[['date', 'chainID', 'quantity', 'styleNumber', 'retailerID']]
    chains = df['chainID'].unique()
    acceleration = pd.DataFrame(columns=['accelerationChainStyle', 'accelerationRetailerStyle'], index=df.index)
    acceleration[['date', 'styleNumber', 'chainID', 'retailerID']] = df[['date', 'styleNumber', 'chainID', 'retailerID']]

    for chain in chains:
        chaindf = df[df.chainID == chain]
        styles = chaindf['styleNumber'].unique()
        chaindata = acceleration[acceleration.chainID == chain]
        for style in styles:
            styledf = chaindf[chaindf.styleNumber == style]
            styledf = styledf.groupby('date', as_index=False).sum(numeric_only=True)
            styledf['chainID'] = chain
            prev = styledf[['date', 'quantity']].shift(periods=1)
            timedifference = (styledf['date'] - prev['date']).dt.days
            timedifference = timedifference.fillna(value=1)
            styleslope = (styledf['quantity'] - prev['quantity']) / timedifference
            styleslope = styleslope.fillna(value=1)
            #Nu vi har en hastighed søger vi en acceleration
            prev = styleslope.shift(periods=1)
            prev = prev.fillna(value=1)
            tempaccel = (styleslope - prev) / timedifference
            styledata = chaindata[chaindata.styleNumber == style]
            for x in tempaccel.index:
                datedata = styledata[styledata.date == styledf['date'].iloc[x]]
                datedata['accelerationChainStyle'] = tempaccel.iloc[x]
                acceleration.update(datedata)

    retailers = df['retailerID'].unique()
    for retailer in retailers:
         retailerdf = df[df.retailerID == retailer]
         styles = retailerdf['styleNumber'].unique()
         retailerdata = acceleration[acceleration.retailerID == retailer]
         for style in styles:
             styledf = retailerdf[retailerdf.styleNumber == style]
             styledf = styledf.groupby('date', as_index=False).sum(numeric_only=True)
             styledf['retailerID'] = retailer
             prev = styledf[['date', 'quantity']].shift(periods=1)
             timedifference = (styledf['date'] - prev['date']).dt.days
             timedifference = timedifference.fillna(value=1)
             styleslope = (styledf['quantity'] - prev['quantity']) / timedifference
             styleslope = styleslope.fillna(value=1)
             # Nu vi har en hastighed søger vi en acceleration
             prev = styleslope.shift(periods=1)
             prev = prev.fillna(value=1)
             tempaccel = (styleslope - prev) / timedifference
             styledata = retailerdata[retailerdata.styleNumber == style]
             for x in styleslope.index:
                 datedata = styledata[styledata.date == styledf['date'].iloc[x]]
                 datedata['accelerationRetailerStyle'] = tempaccel.iloc[x]
                 acceleration.update(datedata)
    acceleration = acceleration.fillna(value=0)
    return acceleration[['accelerationChainStyle', 'accelerationRetailerStyle']]



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

    return data['color_popularity']

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

    def get_size_feature(self, size, chainID, male = False):
        if chainID == 1:
            return self.mapping_ch1[size]
        elif chainID == 2:
            if male:
                return self.mapping_ch2_m[size]
            else:
                return self.mapping_ch2_w[size]
        elif chainID == 3:
            return self.mapping_ch3[size]
        else: 
            return 8888

# requires that chainid and ismale is present
def make_sizefeature_col(df):
    sf = SizeFeature()

    data = df.copy()
    data['size_scale'] = tuple(map(lambda size, chainID, ismale: sf.get_size_feature(size=size, chainID = chainID, male = ismale), data['size'], data['chainID'], data['ismale']))
    return data['size_scale']

def featurize(df, path):
    functionlist = {'quantityChain':today_quantity, 'turnoverChain':today_turnover}#, 'month_feat': month_feature,
                    # 'month_change_dist':month_change_dist_feature, 'month_first_dist':month_first_dist_feature,
                    # 'month_next_first_dist': month_next_first_dist_feature, 'quantity7d':week_quantity_feature,
                    # 'quantity30d':month_quantity_feature, 'weekday':weekday_feature, 'discount_percent':discount_to_percent,
                    # 'avg_price_style':get_avg_price_in_style, 'color':make_feature_col, 'age':featurealder,
                    # 'acceleration':featureacceleration, 'size':make_sizefeature_col,
                    # 'PLCBD':featureplcBD, 'PLCCD':featureplcCD, 'stigning_Periode':stigning_period}


    featuredf = weekday_feature(df)
    for func in functionlist:
        print(func)
        temp = functionlist[func](df)
        print(temp.isnull().values.any())
        if isinstance(temp, pd.DataFrame):
            featuredf[temp.columns] = temp
            print(featuredf.isnull.any())
        if isinstance(temp, pd.Series):
            featuredf[func] = temp
            print(featuredf.isnull().values.any())
    featuredf = featuredf.fillna(value = 9999)
    return featuredf.to_csv(path_or_buf=path + 'Features.rpt', index=False, sep=';', encoding='utf-8')
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#Load clean data
#sm_dir = 'C:/Users/SMSpin/Documents/GitHub/P5/CleanData/CleanedData.rpt'
kloster_dir = r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\'
patrick_dir = r'C:\Users\Patrick\PycharmProjects\untitled\CleanData\\'
ng_dir = r'C:\P5GIT\\'

dataframe = dl.load_sales_file(ng_dir + 'CleanedData.rpt')

dataframetest = dataframe[dataframe.styleNumber == 'Z99319B']
dataframetest = dataframetest.append(dataframe[dataframe.styleNumber == '010668A'])
dataframetest = dataframetest.append(dataframe[dataframe.styleNumber == 'Y95901D'])

# chains = dataframe['chainID'].unique()
# print('chains: {0}'.format(len(chains)))
# for chain in chains:
#     chaindf = dataframe[dataframe.chainID == chain]
#     products = chaindf['productID'].unique()
#     chaindata = dataframe[dataframe.chainID == chain]
#     print('products: {0}'.format(len(products)))
#     for product in products:
#         productdf = chaindf[chaindf.productID == product]
#         productdf = productdf.groupby('date', as_index=False).sum(numeric_only=True)
#         productdata = chaindata[chaindata.productID == product]
#         print('prod by date: {0}'.format(len(productdf)))
#         for x in productdf.index:
#             datedata = productdata[productdata.date == productdf['date'].iloc[x]]
#             datedata['quantityChain'] = productdf['quantity'].iloc[x]
print(dataframetest)
print(feature_age(dataframetest))

# print(featurize(dataframetest, patrick_dir))

# dataframeother = dl.load_feature_file(patrick_dir + 'Features.rpt')

# print(dataframeother)
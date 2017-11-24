import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from calendar import monthrange
import time
import numpy as np
import calendar

# total turnover in distinct periods
def total_turnover_in_period_agg(df, period = 'W-SUN'):
    data = df[['date', 'turnover']].copy()

    turnover_by_date = data.groupby('date').sum()

    # total turnover for previous period.
    summed_turnover = turnover_by_date.resample(period).sum()#.shift(1).fillna(0)

    # reindex to get all dates. values are filled forward
    result = summed_turnover.reindex(turnover_by_date.index, method='pad').fillna(0)

    data['total_turnover'] = tuple(map(lambda date: result.loc[date, 'turnover'], data['date']))
    return data['total_turnover']

# total turnover in rolling periods
def total_turnover_in_period_rolling(df, period = '7D'):
    data = df[['date', 'turnover']].copy()

    turnover_by_date = data.groupby('date').sum().rolling(period).sum()#.shift(1).fillna(0)
    data['total_turnover'] = tuple(map(lambda date: turnover_by_date.loc[date, 'turnover'], data['date']))
    return data['total_turnover']

# total quantity in distinct periods
def total_quantity_in_period_agg(df, period = 'W-SUN'):
    data = df[['date', 'quantity']].copy()

    quantity_by_date = data.groupby('date').sum()
    # total quantity for previous period.
    summed_quantity = quantity_by_date.resample(period).sum()#.shift(1).fillna(0)

    # reindex to get all dates. values are filled forward
    result = summed_quantity.reindex(quantity_by_date.index, method='pad').fillna(0)

    data['total_quantity'] = tuple(map(lambda date: result.loc[date, 'quantity'], data['date']))
    return data['total_quantity']

# total quantity in rolling periods
def total_quantity_in_period_rolling(df, period = '7D'):
    data = df[['date', 'quantity']].copy()

    quantity_by_date = data.groupby('date').sum().rolling(period).sum()#.shift(1).fillna(0)
    data['total_quantity'] = tuple(map(lambda date: quantity_by_date.loc[date, 'quantity'], data['date']))
    return data['total_quantity']


def quantity_in_periods_agg(df, period = 'W-SUN', n = 3, on ='productID'):
    data = df[['date', on, 'quantity']].copy()
    dates = data['date'].unique()

    p = pd.pivot_table(data, values='quantity', index=['date'], columns=on, aggfunc=np.sum).fillna(0).resample(period).sum()
    col_names = []
    for i in range(0,n):
        result = p.shift(i).fillna(0).reindex(dates, method='pad')
        col_name = 'qty_p' + str(i+1)
        col_names.append(col_name)
        data[col_name] = tuple(map(lambda date, label: result.loc[date, label], data['date'], data[on]))

    data[col_names] = data[col_names].fillna(0)
    return data[col_names]

def quantity_in_periods_rolling(df, days = 7, n = 3, on ='productID'):
    data = df[['date', on, 'quantity']].copy()
    dates = data['date'].unique()
    days_s = str(days) + 'D'

    p = pd.pivot_table(data, values='quantity', index=['date'], columns=on, aggfunc=np.sum).fillna(0)
    dates = p.index
    realdates = pd.date_range(dates[0], dates[len(dates)-1])
    p = p.reindex(realdates)
    p = p.rolling(days_s).sum()
    col_names = []
    for i in range(0,n):
        result = p.shift(i * days).fillna(0)
        col_name = 'qty_p' + str(i+1)
        col_names.append(col_name)
        data[col_name] = tuple(map(lambda date, label: result.loc[date, label], data['date'], data[on]))

    data[col_names] = data[col_names].fillna(0)
    return data[col_names]


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
    df = dataframe[['discount','turnover']].copy()

    df['discountP'] = abs(df['discount'])/(abs(df['discount'])+abs(df['turnover']))*100

    return df['discountP']

#Returns a list of avg style price for each transaction
def get_price(dataframe):
    df = dataframe[['styleNumber','turnover','discount','quantity']].copy()

    df['price'] = (abs(df['turnover'])+abs(df['discount'])) / abs(df['quantity'])

    return df['price']

#Returns a list with average day price for each transaction.
#Executive grazt
def create_avg_list(dataframe):
    data = dataframe[['date','turnover','discount','quantity']].copy()
    df = data.groupby('date').sum(numeric_only = True)
    df['avg_day_price'] = (abs(df['turnover'])+abs(df['discount']))/abs(df['quantity'])

    data['avg_price'] = tuple(map(lambda date: df.loc[date, 'avg_day_price'], data['date']))

    return data['avg_price']


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


# laver alder på stylenumber baseret på df
def age_feature(df):
    data = df[['date', 'styleNumber']].copy()

    firstdates = data.groupby('styleNumber').first()

    data['first'] = tuple(map(lambda style: firstdates.loc[style,'date'], data['styleNumber']))
    data['styleage'] = (data['date'] - data['first']).dt.days

    return data['styleage']



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

def make_colorfeature(df, window = '7d'):
    cfp = ColorFeatureProvider(df, window)
    data = df.copy()
    data.colorname = data.colorname.str.lower().str.strip()
    data = data.dropna(axis=0, how='any')

    # data[col_name] = np.nan

    data['color_popularity'] = tuple(map(lambda color, date: cfp.get_color_popularity(color = color, date = date), data['colorname'], data['date']))

    return data['color_popularity'].fillna(-9999)

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
                            'Unknown' : -1}

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
def make_sizefeature(df):
    sf = SizeFeature()

    data = df.copy()
    data['size_scale'] = tuple(map(lambda size, chainID, ismale: sf.get_size_feature(size=size, chainID = chainID, male = ismale), data['size'], data['chainID'], data['ismale']))
    return data['size_scale']

def target_values_agg(df, days = 'W-SUN', on = 'productID'):
    data = df.copy()

    data = data[['date', on, 'quantity']]

    p = pd.pivot_table(data, values='quantity', index=['date'], columns=[on], aggfunc=np.sum).fillna(0)
    p = p.resample(days).sum()
    p = p.shift(-1)
    p = p.fillna(-9999)
    p = p.reindex(data['date'].unique(), method='pad')

    data['target'] = tuple(map(lambda date, label: p.loc[date, label], data['date'], data[on]))

    result = data['target'].fillna(-9999)
    # print(result.value_counts())
    return result

def target_values_rolling(df, days = 7, on = 'productID'):
    data = df.copy()

    data = data[['date', on, 'quantity']]

    days_s = str(days) + 'D'

    p = pd.pivot_table(data, values='quantity', index=['date'], columns=[on], aggfunc=np.sum).fillna(0)
    dates = p.index
    realdates = pd.date_range(dates[0], dates[len(dates)-1])
    p = p.reindex(realdates)
    p = p.rolling(days_s).sum()
    p = p.shift(-days)
    p = p.fillna(-9999)

    data['target'] = tuple(map(lambda date, label: p.loc[date, label], data['date'], data[on]))
    result = data['target'].fillna(-9999)
    # print(result.value_counts())
    return result

def featurize2(df):
    data = df.copy()

    print('making global features...')
    data['size_scale'] = make_sizefeature(data)
    data['discount_pct'] = discount_to_percent(data)
    data['price'] = get_price(data)


    # chain features
    print('making chain-level features...')
    chains = df['chainID'].unique()
    for c in chains:
        print('chain: ' + str(c))
        chain_data = data[data.chainID == c]
        # data.loc[data.chainID == c,'color_popularity_chain'] = make_colorfeature(chain_data, window = '7D')
        data.loc[data.chainID == c,'style_age_chain'] = age_feature(chain_data)
        data.loc[data.chainID == c,'total_turnover_chain_rolling'] = total_turnover_in_period_rolling(chain_data, period = '7D')
        # data.loc[data.chainID == c,'total_turnover_chain_agg'] = total_turnover_in_period_agg(chain_data, period = 'W-SUN')
        data.loc[data.chainID == c,'total_quantity_chain_rolling'] = total_quantity_in_period_rolling(chain_data, period = '7D')
        # data.loc[data.chainID == c,'total_quantity_chain_agg'] = total_quantity_in_period_agg(chain_data, period = 'W-SUN')

        # qty_agg = quantity_in_periods_agg(chain_data, period = 'W-SUN', n = 3, on ='productID')
        # data.loc[data.chainID == c,'qty_p1_chain_prod_agg_sun'] = qty_agg['qty_p1']
        # data.loc[data.chainID == c,'qty_p2_chain_prod_agg_sun'] = qty_agg['qty_p2']
        # data.loc[data.chainID == c,'qty_p3_chain_prod_agg_sun'] = qty_agg['qty_p3']
 
        qty_rolling = quantity_in_periods_rolling(chain_data, days = 7, n = 3, on ='productID')
        data.loc[data.chainID == c,'qty_p1_chain_prod_rolling_7'] = qty_rolling['qty_p1']
        data.loc[data.chainID == c,'qty_p2_chain_prod_rolling_7'] = qty_rolling['qty_p2']
        data.loc[data.chainID == c,'qty_p3_chain_prod_rolling_7'] = qty_rolling['qty_p3']

        data.loc[data.chainID == c,'avg_price_chain']  = create_avg_list(chain_data)

    # retailer features
    print('making retailer-level features...')
    retailers = df['retailerID'].unique()
    l = len(retailers)
    n = 1

    for r in retailers:
        print('retailer: ' + str(r)+ ' - ' + str(n) + ' of ' + str(l))
        n += 1

        retailer_data = data[data.retailerID == r]

        # data.loc[data.retailerID == r,'color_popularity_ret'] = make_colorfeature(retailer_data, window = '7D')
        data.loc[data.retailerID == r,'style_age_ret'] = age_feature(retailer_data)
        data.loc[data.retailerID == r,'total_turnover_ret_rolling_7'] = total_turnover_in_period_rolling(retailer_data, period = '7D')
        # data.loc[data.retailerID == r,'total_turnover_ret_agg_sun'] = total_turnover_in_period_agg(retailer_data, period = 'W-SUN')
        data.loc[data.retailerID == r,'total_quantity_ret_rolling_7'] = total_quantity_in_period_rolling(retailer_data, period = '7D')
        # data.loc[data.retailerID == r,'total_quantity_ret_agg_sun'] = total_quantity_in_period_agg(retailer_data, period = 'W-SUN')

        data.loc[data.retailerID == r,'avg_price_ret'] = create_avg_list(retailer_data)

        qty_rolling = quantity_in_periods_rolling(retailer_data, days = 7, n = 3, on ='productID')
        data.loc[data.retailerID == r,'qty_p1_ret_prod_rolling_7'] = qty_rolling['qty_p1']
        data.loc[data.retailerID == r,'qty_p2_ret_prod_rolling_7'] = qty_rolling['qty_p2']
        data.loc[data.retailerID == r,'qty_p3_ret_prod_rolling_7'] = qty_rolling['qty_p3']

        # qty_agg = quantity_in_periods_agg(retailer_data, period = 'W-SUN', n = 3, on ='productID')
        # data.loc[data.retailerID == r,'qty_p1_ret_prod_agg_sun'] = qty_agg['qty_p1']
        # data.loc[data.retailerID == r,'qty_p2_ret_prod_agg_sun'] = qty_agg['qty_p2']
        # data.loc[data.retailerID == r,'qty_p3_ret_prod_agg_sun'] = qty_agg['qty_p3']

        # retaile data opdateres her
        # retailer_data = data[data.retailerID == r]

        # data.loc[data.retailerID == r,'slope_p1p2'] = retailer_data['qty_p1'] - retailer_data['qty_p2']
        # data.loc[data.retailerID == r,'slope_p2p3'] = retailer_data['qty_p2'] - retailer_data['qty_p3']

        # retailer_data = data[data.retailerID == r]

        # data.loc[data.retailerID == r,'acc_p1p3'] = retailer_data['slope_p1p2'] - retailer_data['slope_p2p3']

        # data.loc[data.retailerID == r, 'target_prod_agg_sun'] = target_values_agg(retailer_data, days = 'W-SUN', on = 'productID')
        data.loc[data.retailerID == r, 'target_prod_rolling_7'] = target_values_rolling(retailer_data, days = 7, on = 'productID')

        
    # print(data['target'].tail(10))


    # print(str(len(data)))
    # result = result[result.target_prod_rolling_7 != -9999]
    # print(str(len(result)))

    data['qty_speed_ret_prod_p1p2'] = data['qty_p1_ret_prod_rolling_7'] -  data['qty_p2_ret_prod_rolling_7']
    data['qty_speed_ret_prod_p2p3'] = data['qty_p2_ret_prod_rolling_7'] -  data['qty_p3_ret_prod_rolling_7']
    data['qty_acc_ret_prod_p1p3'] = data['qty_speed_ret_prod_p1p2'] -  data['qty_speed_ret_prod_p2p3']


    return data


#Load clean data
#sm_dir = 'C:/Users/SMSpin/Documents/GitHub/P5/CleanData/CleanedData.rpt'
kloster_dir = r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\'
patrick_dir = r'C:\Users\Patrick\PycharmProjects\untitled\CleanData\\'
ng_dir = r'C:\P5GIT\\'

print('Loading data...')

dataframe = dl.load_sales_file(ng_dir + 'CleanedData_no_isnos_no_outliers.rpt')
dataframe = dataframe.dropna(axis=0, how='any')

rets = [3,4]

# mysample = dataframe.sample(10000, random_state = 1234)

mysample = dataframe[dataframe.retailerID.isin(rets)]

testframe = featurize2(dataframe[dataframe.chainID == 1])

testframe.to_csv('featurized_ch1_new.csv', na_rep = 'bananaphone', index = False, sep=';',encoding='utf-8')

# import linreg

# features = [
#     'qty_p1', 
#     'slope_p1p2' , 
#     # 'acc_p1p3' ,
#     'target']

# linreg.regress(testframe, features)

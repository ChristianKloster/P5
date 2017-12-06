import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from sklearn.linear_model import LinearRegression
import sklearn.metrics #import mean_squared_error, r2_score
import targetprovider
import os
# import FeaturerizerOld as FO
import Featurerizer as FO
import math
import linreg
from sklearn.metrics import mean_squared_error as mse

def naive_model_smart(df, days = 7, on = 'productID'):
    data = df.copy()

    data = data[['date', on, 'quantity']]

    days_s = str(days) + 'D'

    p = pd.pivot_table(data, values='quantity', index=['date'], columns=[on], aggfunc=np.sum).fillna(0)
    dates = p.index
    realdates = pd.date_range(dates[0], dates[len(dates) - 1])
    p = p.reindex(realdates)
    p = p.rolling(days_s).sum()
    target = p.shift(-days)
    target = target.fillna(0)

    data['Previous'] = tuple(map(lambda date, label: p.loc[date, label], data['date'], data[on]))
    data['Next'] = tuple(map(lambda date, label: target.loc[date, label], data['date'], data[on]))

    errorDF = pd.DataFrame(columns=[['SSE', 'RMS', 'MSE', 'MaxError']], index=[0])

    errorDF['SSE'] = sum_square_error(data['Previous'], data['Next'])
    errorDF['RMS'] = root_mean_squares(data['Previous'], data['Next'])
    errorDF['MSE'] = mse(data['Previous'], data['Next'])
    errorDF['MaxError'] = max(abs(data['Previous'] - data['Next']))

    return errorDF


def naivemodel(today, tomorrow):
    #Giver fejlen i prediction i tøjmængde, negativ er for lidt forudset, positiv for meget forudset
    error = tomorrow - today
    lifetime = today.copy() #pd.DataFrame(data = today, index=today.index)
    lifetime[::] = np.nan
    test = ~today.isin([0])
    for c in today.columns:
        nozero = today[c][test[c]]
        if nozero.size == 0:
            break
        elif nozero.size == 1:
            first = nozero.index
            last = first
            errorreal = tomorrow[c].loc[first[0]:last[0]] - today[c].loc[first[0]:last[0]]
        else:
            first = nozero.first('1W').index
            last = nozero.last('1W').index
            errorreal = tomorrow[c].loc[first[0]:last[0]] - today[c].loc[first[0]:last[0]]
        lifetime[c] = errorreal
    errormargin = lifetime.apply(pd.value_counts).fillna(0)

    if 0 in errormargin.index._data:
        percent = (errormargin.loc[0]/errormargin.sum(axis=0)) * 100
        percent = percent.sum()
    else:
        percent = 0
    errormargin = errormargin.sum(axis=1)
    return lifetime, errormargin, percent

def test_of_model(today, predicted):
    # Giver fejlen i prediction i tøjmængde, negativ er for lidt forudset, positiv for meget forudset
    testtarget = today.copy()
    # testtarget = testtarget.iloc[::-1]
    # testtarget = testtarget.head(int(np.ceil(testtarget.shape[0] * (1-splitpercent))))
    # testtarget = testtarget.iloc[::-1]
    predictedround = np.round_(predicted)
    error = predictedround - testtarget
    error = pd.Series(error)

    errormargin = error.value_counts().fillna(0)
    # print('fejl margen')
    # print(errormargin)
    if 0 in errormargin.index._data:
        percent = (errormargin.loc[0]/errormargin.sum(axis=0)) * 100
        percent = percent.sum()
    else:
        percent = 0

    # print('Procent')
    # print(percent)
    # print(percent.describe())
    return error, errormargin, percent

def sum_square_error(actual, predict):
    difference = actual-predict
    SE = difference**2
    SSE = SE.sum()
    return SSE

def root_mean_squares(actual, predict):
    SSE = sum_square_error(actual, predict)
    RMS = math.sqrt(SSE/actual.size)
    return RMS

def tester(df, featuredf, number):
    # retailers = [2, 4]
    mysample = df.copy()#df[df.retailerID.isin([3,4])]  # .sample(50000, random_state = 1234)

    testframe = featuredf.copy()#FO.featurize2(df)

    features = [
        # 'size_scale',
        'discount_pct',
        'price',
        # 'style_age_chain',
        # 'total_turnover_chain_rolling',
        'total_quantity_chain_rolling',
        'qty_p1_chain_prod_rolling_7',
        'qty_p2_chain_prod_rolling_7',
        # 'qty_p3_chain_prod_rolling_7',
        # 'avg_price_chain',
        'style_age_ret',
        # 'total_turnover_ret_rolling_7',
        'total_quantity_ret_rolling_7',
        # 'avg_price_ret',
        'qty_p1_ret_prod_rolling_7',
        'qty_p2_ret_prod_rolling_7',
        # 'qty_p3_ret_prod_rolling_7',
        'qty_speed_ret_prod_p1p2',
        # 'qty_speed_ret_prod_p2p3',
        # 'qty_acc_ret_prod_p1p3'
    ]

    predict = linreg.regress(testframe, features, target = 'target_prod_rolling_7')
    generer_fra = 'productID'
    #Til et længere tid men mere automatiseret
    retailers = mysample.retailerID.unique()
    products = mysample.productID.unique()
    errormargin = pd.Series([0, 0, 0])
    percent = 0
    test = test_of_model(predict[2], predict[1])
    for ret in retailers:
        # baseline
        quantity1W = targetprovider.get_pivot_tables_with_target_values(df, retailerid=ret, days='W-SUN', on=generer_fra)
        # splitting into target and feature, ie today and tomorrow
        traintarget = quantity1W[0]
        trainfeature = quantity1W[1]
        week = naivemodel(trainfeature, traintarget)
        percent = percent + week[2]

        errormargin = errormargin.add(week[1], fill_value=0)
    percent = percent / products.size
    errormarginmodel = test[1]
    errormarginmodel = errormarginmodel.sort_index()
    print('Naiv: {0} Model: {1} Forbedring: {2}'.format(percent, test[2], test[2] - percent))
    print('Naiv fejlmargener')
    print(errormargin)
    print('Model fejlmargener')
    print(errormarginmodel)

    # appendname = 'linreg'
    directory = os.path.dirname(r'C:\Users\Patrick\PycharmProjects\untitled\2x80NNsgd\\')
    if not os.path.exists(directory):
        os.makedirs(directory)

    errormargin = errormargin / errormargin.max()#products.size
    # errormargin = errormargin.drop(axis=0, labels=0)
    plt.figure()
    errormargin.plot()
    plt.ylabel('Hyppighed')
    plt.xlabel('Fejl')
    plt.title('Naiv fejl margin, procent {0}'.format(percent))
    plt.tight_layout()
    plt.savefig('{0}/naiv_error_margin_norm_{1}'.format(directory,number))

    plt.close()
    errormarginmodel = errormarginmodel / errormarginmodel.max()#products.size
    # errormarginmodel = errormarginmodel.drop(axis=0, labels=0)
    plt.figure()
    errormarginmodel.plot()
    plt.ylabel('Hyppighed')
    plt.xlabel('Fejl')
    plt.title('Model error margin, procent {0}'.format(test[2]))
    plt.tight_layout()
    plt.savefig('{0}/Model_error_margin_norm_{1}'.format(directory,number))

    plt.close()
    pred = predict[1]
    base = predict[2]
    plt.figure()
    split = round(len(mysample) * 0.9)
    difference = pred.shape[0] - mysample[split:].shape[0]
    mytestsample = mysample[split - difference:]
    mytestsample['pred'] = np.round_(pred)
    mytestsample['base'] = base
    plt.plot(mysample['date'][split - difference:] , pred, '-', color = 'red')
    plt.plot(mysample['date'][split - difference:] , base, '-', color = 'blue')
    plt.ylabel('quantity')
    plt.xlabel('date')
    plt.title('Model og Predict salg')
    plt.tight_layout()
    plt.savefig('{0}/Model_VS_Predict_{1}'.format(directory, number))


kloster_dir = r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\'
patrick_dir = r'C:\Users\Patrick\PycharmProjects\untitled\CleanData\\'
ng_dir = r'C:\P5GIT\\'
print('Loading data...')
# dataframe = dl.load_sales_file(patrick_dir + 'CleanedData.rpt')

dataframe = dl.load_sales_file(patrick_dir + 'CleanedData_no_isnos_no_outliers.rpt')

features = [
    # 'size_scale',
    # 'discount_pct',
    # 'price',
    # 'style_age_chain',
    # 'total_turnover_chain_rolling',
    # 'total_quantity_chain_rolling',
    # 'qty_p1_chain_prod_rolling_7',
    # 'qty_p2_chain_prod_rolling_7',
    # 'qty_p3_chain_prod_rolling_7',
    # # 'avg_price_chain',
    # 'style_age_ret',
    # 'total_turnover_ret_rolling_7',
    # 'total_quantity_ret_rolling_7',
    # 'avg_price_ret',
    'qty_p1_ret_prod_rolling_7',
    'qty_p2_ret_prod_rolling_7',
    'qty_p3_ret_prod_rolling_7',
    'qty_speed_ret_prod_p1p2_rolling_7',
    'qty_speed_ret_prod_p2p3_rolling_7',
    'qty_acc_ret_prod_p1p3_rolling_7',
    'qty_p1_ret_prod_rolling_3',
    'qty_p2_ret_prod_rolling_3',
    'qty_p3_ret_prod_rolling_3',
    'qty_speed_ret_prod_p1p2_rolling_3',
    'qty_speed_ret_prod_p2p3_rolling_3',
    'qty_acc_ret_prod_p1p3_rolling_3',
    'qty_p1_ret_prod_rolling_1',
    'qty_p2_ret_prod_rolling_1',
    'qty_p3_ret_prod_rolling_1',
    'qty_speed_ret_prod_p1p2_rolling_1',
    'qty_speed_ret_prod_p2p3_rolling_1',
    'qty_acc_ret_prod_p1p3_rolling_1',

    'qty_p1_ret_style_rolling_7',
    'qty_p2_ret_style_rolling_7',
    'qty_p3_ret_style_rolling_7',
    # 'qty_speed_ret_style_p1p2_rolling_7',
    # 'qty_speed_ret_style_p2p3_rolling_7',
    # 'qty_acc_ret_style_p1p3_rolling_7',
    'qty_p1_ret_style_rolling_3',
    'qty_p2_ret_style_rolling_3',
    'qty_p3_ret_style_rolling_3',
    #'qty_speed_ret_style_p1p2_rolling_3',
    #'qty_speed_ret_style_p2p3_rolling_3',
    #'qty_acc_ret_style_p1p3_rolling_3',
    'qty_p1_ret_style_rolling_1',
    'qty_p2_ret_style_rolling_1',
    'qty_p3_ret_style_rolling_1',
    # 'qty_speed_ret_style_p1p2_rolling_1',
    # 'qty_speed_ret_style_p2p3_rolling_1',
    # 'qty_acc_ret_style_p1p3_rolling_1'
    # 'target_prod_agg_sun'
    'target_prod_rolling_7'
    ]
#Fast approach

n = 1
dppp = 0
dpp = 0
dp = 0
dataframe = dataframe[dataframe.chainID == 1]
chains = dataframe.chainID.unique()
for chain in chains:
    chainframe = dataframe[dataframe.chainID == chain]
    chainframe = FO.featurize2(chainframe)

    chainframe = chainframe.set_index('date')


    directory = os.path.dirname(r'C:\Users\Patrick\PycharmProjects\untitled\RollingLin{0}\\'.format(chain))
    if not os.path.exists(directory):
         os.makedirs(directory)

    errorDF = pd.DataFrame(columns=[['SSE', 'RMS', 'MSE', 'MaxError']], index=range(0,69))
    n = 1
    for group_name, df_group in chainframe.groupby(pd.TimeGrouper(freq="W")):
        if n == 1:
            df = df_group.copy()
            df = df.reset_index()
        else:
            newframe = df_group.copy().reset_index()
            newframe.index = newframe.index+df._stat_axis.size
            predict = linreg.regress_use_case(df, newframe, features, target='target_prod_rolling_7')
            df = df.append(df_group.copy().reset_index(), ignore_index=True)
            test = test_of_model(predict[2], predict[1])
            errormarginmodel = test[1]
            errormarginmodel = errormarginmodel.sort_index()
            predicted = predict[1]
            predicted = np.round_(predicted)
            plt.close()
            errormarginmodel = errormarginmodel / errormarginmodel.sum()
            plt.figure()
            errormarginmodel.plot()
            plt.ylabel('Hyppighed')
            plt.xlabel('Fejl')
            plt.title('Model error margin, procent {0}'.format(test[2]))
            plt.tight_layout()
            plt.savefig('{0}/Model_error_margin_norm_{1}'.format(directory, n))
            errorDF.iloc[n-1]['SSE'] = sum_square_error(predict[2],predicted)
            errorDF.iloc[n-1]['RMS'] = root_mean_squares(predict[2], predicted)
            errorDF.iloc[n - 1]['MSE'] = mse(predict[2], predicted)
            errorDF.iloc[n - 1]['MaxError'] = max(abs(predict[2] - predicted))
        n = n + 1
        print(errorDF)
    errorDF.to_csv(path_or_buf=directory + 'ErrorFrame.rpt', index=False, sep=';', encoding='utf-8')
    chainframe = chainframe.reset_index()
    naive = naive_model_smart(chainframe)
    naive.to_csv(path_or_buf=directory + 'Naive.rpt', index=False, sep=';', encoding='utf-8')


    #Slow approach from here
# for chain in chains:
#     chainframe = dataframe[dataframe.chainID == chain]
#
#     chainframe = chainframe.set_index('date')
#     errorDF = pd.DataFrame(columns=[['SSE', 'RMS', 'MSE', 'MaxError']], index=range(0,69))
#     directory = os.path.dirname(r'C:\Users\Patrick\PycharmProjects\untitled\NBChain{0}NFReplaceW1\\'.format(chain))
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     for group_name, df_group in chainframe.groupby(pd.TimeGrouper(freq="W")):
#         if n == 1:
#             df = df_group.copy()
#             df = df.reset_index()
#             featureframe = FO.featurize2(df)
#         else:
#             df = df.append(df_group.copy().reset_index(), ignore_index=True)
#             if dppp != 0:
#                 mask = (df['date'] > dppp)
#                 newfeature = df.loc[mask]
#                 if not(newfeature.empty):
#                     newframe = FO.featurize2(newfeature)  # df_group.copy().reset_index())
#                     featureframe = featureframe.append(newframe)
#                     newframe.index = newframe.index+df._stat_axis.size
#             elif dpp != 0:
#                 mask = (df['date'] > dpp)
#                 newfeature = df.loc[mask]
#                 if not(newfeature.empty):
#                     newframe = FO.featurize2(newfeature)#df_group.copy().reset_index())
#                     featureframe = featureframe.append(newframe)
#                     newframe.index = newframe.index+df._stat_axis.size
#             elif dp != 0:
#                 newfeature = df_group.copy().reset_index()
#                 if not(newfeature.empty):
#                     newframe = FO.featurize2(newfeature)  # df_group.copy().reset_index())
#                     featureframe = featureframe.append(newframe)
#                     newframe.index = newframe.index + df._stat_axis.size
#             if not(featureframe.empty):
#                 predict = linreg.regress_use_case(featureframe, newframe, features, target='target_prod_rolling_7')
#             test = test_of_model(predict[2], predict[1])
#             errormarginmodel = test[1]
#             errormarginmodel = errormarginmodel.sort_index()
#             predicted = predict[1]
#             predicted = np.round_(predicted)
#             plt.close()
#             errormarginmodel = errormarginmodel / errormarginmodel.sum()
#             plt.figure()
#             errormarginmodel.plot()
#             plt.ylabel('Hyppighed')
#             plt.xlabel('Fejl')
#             plt.title('Model error margin, procent {0}'.format(test[2]))
#             plt.tight_layout()
#             plt.savefig('{0}/Model_error_margin_norm_{1}'.format(directory, n))
#             errorDF.iloc[n - 1]['SSE'] = sum_square_error(predict[2], predicted)
#             errorDF.iloc[n - 1]['RMS'] = root_mean_squares(predict[2], predicted)
#             errorDF.iloc[n - 1]['MSE'] = mse(predict[2], predicted)
#             errorDF.iloc[n - 1]['MaxError'] = max(abs(predict[2] - predicted))
#         n = n + 1
#         dppp = dpp
#         dpp = dp
#         dp = group_name
#     print(errorDF)
#     errorDF = errorDF.fillna(value=0)
#     errorDF.to_csv(path_or_buf=directory + 'ErrorFrame1.rpt', index=False, sep=';', encoding='utf-8')
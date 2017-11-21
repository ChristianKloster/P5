import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from sklearn.linear_model import LinearRegression
import sklearn.metrics #import mean_squared_error, r2_score
import targetprovider
import FeaturerizerOld as FO
import linreg


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
    # print('fejl margen')
    # print(errormargin) .dropna()
    percent = (errormargin.loc[0]/errormargin.sum(axis=0)) * 100
    averagepercent = percent.sum()/percent.size
    errormargin = errormargin.sum(axis=1)
    # Without regard for lifetime
    # errormargin = error.apply(pd.value_counts).fillna(0)
    # percent = (errormargin.loc[0]/errormargin.sum(axis=0)).dropna() * 100
    # averagepercent = percent.sum()/percent.size
    # print('Procent')
    # print(percent)
    # print('Average procent')
    # print(averagepercent)
    # print(percent.describe())
    return lifetime, errormargin, percent, averagepercent

def test_of_model(today, predicted, splitpercent):
    # Giver fejlen i prediction i tøjmængde, negativ er for lidt forudset, positiv for meget forudset
    testtarget = today.copy()
    testtarget = testtarget.iloc[::-1]
    testtarget = testtarget.head(int(np.floor(testtarget.shape[0] * (1-splitpercent))))
    testtarget = testtarget.iloc[::-1]
    predictedround = np.round_(predicted)
    error = predictedround - testtarget['target']

    errormargin = error.value_counts().fillna(0)
    # print('fejl margen')
    # print(errormargin)
    percent = (errormargin.loc[0] / errormargin.sum(axis=0)) * 100

    # print('Procent')
    # print(percent)
    # print(percent.describe())
    return error, errormargin, percent

def tester(df):
    ret = 4
    mysample = df[df.retailerID == ret]  # .sample(50000, random_state = 1234)

    testframe = FO.featurize2(mysample)
    # df, col_names = quantity_in_periods_agg(dataframe)

    features = [
        'qty_p1',
        'slope_p1p2',
        # 'acc_p1p3' ,
        'target']

    predict = linreg.regress(testframe, features)
    generer_fra = 'productID'
    #Til et længere tid men mere automatiseret
    # retailers = mysample.retailerID.unique()
    # errormargin = 0
    # percent = 0

    # for ret in retailers:
    # baseline
    quantity1W = targetprovider.get_pivot_tables_with_target_values(dataframe, retailerid=ret, days='W-SUN', on=generer_fra)
    # splitting into target and feature, ie today and tomorrow
    traintarget = quantity1W[0]
    trainfeature = quantity1W[1]

    week = naivemodel(trainfeature, traintarget)
    # percent += week[3]
    # errormargin += week[1]

    # percent = percent/retailers.size
    test = test_of_model(testframe, predict[0], predict[2])
    print('Procent rigtig')
    print('Naiv: {0} Model: {1} Forbedring: {2}'.format(week[3], test[2], test[2] - week[3]))
    print('Naiv fejlmargener')
    print(week[1])
    #længere tid men mere auto
    # print('Naiv: {0} Model: {1} Forbedring: {2}'.format(percent, test[2], test[2] - percent))
    # print('Naiv fejlmargener')
    # print(errormargin)
    print('Model fejlmargener')
    print(test[1])

kloster_dir = r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\'
patrick_dir = r'C:\Users\Patrick\PycharmProjects\untitled\CleanData\\'
ng_dir = r'C:\P5GIT\\'

dataframe = dl.load_sales_file(patrick_dir + 'CleanedData.rpt')
tester(dataframe)

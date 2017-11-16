import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from sklearn.linear_model import LinearRegression
import sklearn.metrics #import mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import targetprovider

def baselinemodeler(targetTrain, featureTrain, testTarget, testFeature):
    # Setting up prediction columns
    target = targetTrain.transpose().fillna(value=0)
    feature = featureTrain.transpose().fillna(value=0)
    columns = testTarget.columns.tolist()
    results = pd.DataFrame(columns=[['Rsquare', 'lin_mse']], index=target.index)
    resultsrelationactual = pd.DataFrame(columns=[columns])
    # resultsrelationactual = resultsrelationactual.fillna(value=0)
    # results['Rsquare'] = 9999
    # results['lin_mse'] = 9999
    dateindex = testTarget.transpose().index

    for product in results.index:
        # Initialize the model class.
        lin_model = LinearRegression(n_jobs=-1)
        # Fit the model to the training data.
        feat = feature.loc[product].values.reshape(-1, 1)
        targ = target.loc[product].values.reshape(-1, 1)
        testF = testFeature.loc[product]
        testF = testF.values.reshape(-1,1)
        testT = testTarget.loc[product]
        testT = testT.values
        lin_model.fit(feat, targ)
        pred = lin_model.predict(testF)

        # print("Predictions:", newpred)
        # Compute error between our test predictions and the actual values
        r2 = sklearn.metrics.r2_score(testT, pred)
        # r2 = sklearn.metrics.accuracy_score(testT, pred)
        # r2 = sklearn.metrics.average_precision_score(testT, pred)
        # r2 = sklearn.metrics.precision_score(testT, pred)
        lin_mse = sklearn.metrics.mean_squared_error(testT, pred)
        results['Rsquare'].loc[product] = r2
        results['lin_mse'].loc[product] = lin_mse
        dumb = testT - pred
        dumb = pd.Series(dumb[0], index=dateindex)
        resultsrelationactual.append(dumb, ignore_index=True) #resultsrelationactual.loc[product] + dumb
        # lin_model =  tuple(map(lambda features, targets : LinearRegression().fit(features, targets), feature[columns], target[columns]))
        # The coefficients
        # print('Coefficients for {1}: {0}'.format(lin_model.coef_, product))

    return  resultsrelationactual, results

def timemodeler(ts, Id, lag = 21):
    ts_log = np.log(ts)
    # ts_log = ts
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    test_stationarity(ts, Id)
    lag_acf = acf(ts_log_diff, nlags=lag, fft=True)
    lag_pacf = pacf(ts_log_diff, nlags=lag, method='ols')
    plt.close()
    # Plot AutoCorrelation Function:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')
    # Plot Partial AutoCorrelation Function:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.savefig('Partial Autocorrelation Function.png')
    plt.close()
    #modeling #AR model
    model = ARIMA(ts_log, order=(int(round(lag_acf[0])), 1, 0))
    results_AR = model.fit(disp=-1)
    plt.plot(ts_log_diff)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('RSS: {0}'.format(sum((results_AR.fittedvalues - ts_log_diff) ** 2)))
    plt.savefig('RSS 1.png'.format(sum((results_AR.fittedvalues - ts_log_diff) ** 2)))
    plt.close()
    #MA model
    model = ARIMA(ts_log, order=(0, 1, int(round(lag_pacf[0]))))
    results_MA = model.fit(disp=-1)
    plt.plot(ts_log_diff)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('RSS: {0}'.format(sum((results_MA.fittedvalues - ts_log_diff) ** 2)))
    plt.savefig('RSS 2.png'.format(sum((results_MA.fittedvalues - ts_log_diff) ** 2)))
    plt.close()
    #Combining models
    if int(round(lag_acf[0])) == int(round(lag_pacf[0])):
        model = ARIMA(ts_log, order=(int(round(lag_acf[0])), int(round(lag_acf[0]))-1, int(round(lag_pacf[0]))))
    else:
        model = ARIMA(ts_log, order=(int(round(lag_acf[0])), 0, int(round(lag_pacf[0]))))
    results_ARIMA = model.fit(disp=-1)
    plt.plot(ts_log_diff)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: {0}'.format(sum((results_ARIMA.fittedvalues - ts_log_diff) ** 2)))
    plt.savefig('RSS 3.png'.format(sum((results_ARIMA.fittedvalues - ts_log_diff) ** 2)))
    #Prediction part
    #Scaling out of log
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    print(predictions_ARIMA_diff.head())
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print(predictions_ARIMA_diff_cumsum.head())
    #Base value af første indgang og summere op med nye værdier
    predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    print(predictions_ARIMA_log.head())
    #Genfinder og sammenligner med original serie
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.close()
    plt.plot(ts)
    plt.plot(predictions_ARIMA)
    plt.title('RMSE: {0}'.format(np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts))))
    plt.savefig('compare{0}-1.png'.format(lag))

def test_stationarity(timeseries, Id):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=3,center=False).mean()
    rolstd = timeseries.rolling(window=3,center=False).std()

    # Plot rolling statistics:
    plt.close()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.savefig("timeindependence{0}.pdf".format(Id))

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def lin_modeller(df):
    pass

def tester():
    pass

kloster_dir = r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\'
ng_dir = r'C:\P5GIT\\'
patrick_dir = r'C:\Users\Patrick\PycharmProjects\untitled\CleanData\\'

dataframe = dl.load_sales_file(patrick_dir + 'CleanedData.rpt')

# dataframe = dataframe[dataframe.retailerID == 2]
# dataframe = dataframe[dataframe.isNOS != 1]
generer_fra = 'productID'

# products = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

#Different test cases for days
quantity1D = targetprovider.get_pivot_tables_with_target_values(dataframe, retailerid = 4, days = '1D', on = generer_fra)
quantity2D = targetprovider.get_pivot_tables_with_target_values(dataframe, retailerid = 4, days = '2D', on = generer_fra)
quantity4D = targetprovider.get_pivot_tables_with_target_values(dataframe, retailerid = 4, days = '4D', on = generer_fra)
quantity1W = targetprovider.get_pivot_tables_with_target_values(dataframe, retailerid = 4, days = '1W', on = generer_fra)
#Creating train data from cases, use whichever one is wanted for testing
traintarget = quantity1D[0]
trainfeature = quantity1D[1]
traintarget = traintarget.head(int(np.floor(traintarget.shape[0] * 0.9)))
trainfeature = trainfeature.head(int(np.floor(trainfeature.shape[0] * 0.9)))
#Creates the test data for verification
testtarget = quantity1D[0].iloc[::-1]
testfeature = quantity1D[1].iloc[::-1]
testtarget = testtarget.head(int(np.ceil(testtarget.shape[0] * 0.1))).iloc[::-1]
testfeature = testfeature.head(int(np.ceil(testfeature.shape[0] * 0.1))).iloc[::-1]
testtarget = testtarget.transpose().fillna(value=0)
testfeature = testfeature.transpose().fillna(value=0)
# Til at teste på udsnit af dataet
# columns = testtarget.columns.tolist()
# traintarget = traintarget.iloc[:, [0,1]]
# trainfeature = trainfeature.iloc[:, [0,1]]
# testtarget = testtarget.iloc[[0,1]]
# testfeature = testfeature.iloc[[0,1]]
daymodel = baselinemodeler(traintarget, trainfeature, testtarget, testfeature)

#Creating train data from cases, use whichever one is wanted for testing
traintarget = quantity2D[0]
trainfeature = quantity2D[1]
traintarget = traintarget.head(int(np.floor(traintarget.shape[0] * 0.9)))
trainfeature = trainfeature.head(int(np.floor(trainfeature.shape[0] * 0.9)))
#Creates the test data for verification
testtarget = quantity2D[0].iloc[::-1]
testfeature = quantity2D[1].iloc[::-1]
testtarget = testtarget.head(int(np.ceil(testtarget.shape[0] * 0.1))).iloc[::-1]
testfeature = testfeature.head(int(np.ceil(testfeature.shape[0] * 0.1))).iloc[::-1]
testtarget = testtarget.transpose().fillna(value=0)
testfeature = testfeature.transpose().fillna(value=0)

twodaymodel = baselinemodeler(traintarget, trainfeature, testtarget, testfeature)

#Creating train data from cases, use whichever one is wanted for testing
traintarget = quantity4D[0]
trainfeature = quantity4D[1]
traintarget = traintarget.head(int(np.floor(traintarget.shape[0] * 0.9)))
trainfeature = trainfeature.head(int(np.floor(trainfeature.shape[0] * 0.9)))
#Creates the test data for verification
testtarget = quantity4D[0].iloc[::-1]
testfeature = quantity4D[1].iloc[::-1]
testtarget = testtarget.head(int(np.ceil(testtarget.shape[0] * 0.1))).iloc[::-1]
testfeature = testfeature.head(int(np.ceil(testfeature.shape[0] * 0.1))).iloc[::-1]
testtarget = testtarget.transpose().fillna(value=0)
testfeature = testfeature.transpose().fillna(value=0)

fourdaymodel = baselinemodeler(traintarget, trainfeature, testtarget, testfeature)

#Creating train data from cases, use whichever one is wanted for testing
traintarget = quantity1W[0]
trainfeature = quantity1W[1]
traintarget = traintarget.head(int(np.floor(traintarget.shape[0] * 0.9)))
trainfeature = trainfeature.head(int(np.floor(trainfeature.shape[0] * 0.9)))
#Creates the test data for verification
testtarget = quantity1W[0].iloc[::-1]
testfeature = quantity1W[1].iloc[::-1]
testtarget = testtarget.head(int(np.ceil(testtarget.shape[0] * 0.1))).iloc[::-1]
testfeature = testfeature.head(int(np.ceil(testfeature.shape[0] * 0.1))).iloc[::-1]
testtarget = testtarget.transpose().fillna(value=0)
testfeature = testfeature.transpose().fillna(value=0)

weekmodel = baselinemodeler(traintarget, trainfeature, testtarget, testfeature)

variances = daymodel[1]
variances['RsquareDay'] = variances['Rsquare']
variances['lin_mseDay'] = variances['lin_mse']
variances['Rsquare2Day'] = twodaymodel[1]['Rsquare']
variances['lin_mse2Day'] = twodaymodel[1]['lin_mse']
# variances['Rsquare4Day'] = fourdaymodel['Rsquare']
# variances['RsquareWeek'] = weekmodel['Rsquare']

print(variances[['RsquareDay', 'Rsquare2Day','lin_mseDay', 'lin_mse2Day']])


# print("Predictions:", newpred)
# Compute error between our test predictions and the actual values
# lin_mse = mean_squared_error(newpred, test['quantity'])
# print("Computed error:", lin_mse)
# print('Variance score: %.2f' % r2_score(test['quantity'], newpred))
# results.iloc[number-1].set_value('productID', a.index[number])
# results.iloc[number-1].set_value('variance', r2_score(test['quantity'], newpred))
# results.iloc[number-1].set_value('error', lin_mse)
# print(results)

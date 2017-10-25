import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoad as dl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


def linearmodeler(d):
    # Setting up prediction columns
    columns = d.columns.tolist()
    columns = [c for c in columns if c not in ['date', 'size', 'SupplierItemgroupName', 'styleNumber', 'colorname',
                                               'isNOS', 'styleNumber', 'description']]
    # Store the variable we'll be predicting on.
    target = "quantity"
    # Generate the training set.  Set random_state to be able to replicate results.
    train = d.sample(frac=0.8, random_state=1)
    # Select anything not in the training set and put it in the testing set.
    test = d.loc[~d.index.isin(train.index)]
    # Initialize the model class.
    lin_model = LinearRegression()
    # Fit the model to the training data.
    lin_model.fit(train[columns], train[target])
    # Generate our predictions for the test set.
    lin_predictions = lin_model.predict(test[columns])
    print("Predictions:", lin_predictions)
    # Compute error between our test predictions and the actual values
    lin_mse = mean_squared_error(lin_predictions, test[target])
    print('Coefficients: \n', lin_model.coef_)
    print("Computed error:", lin_mse)
    print('Variance score: %.2f' % r2_score(test[target], lin_predictions))
    return(lin_model)

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

def tester():
    pass

directory = 'C:/Users/Patrick/PycharmProjects/untitled/AAU/Sales_20'
files = ['1606', '1607', '1608', '1609','1610', '1611', '1612',
         '1701', '1702', '1703', '1704', '1705', '1706' , '1707', '1708', '1709']
end = '.rpt'

for x in range(0,len(files)):
	files[x] = directory + files[x] + end

d = dl.loadSalesFiles(files)
d = d.dropna(axis=0, how='any')
d = d[d.isNOS != 1]
d = d[d.retailerID == 42]
generer_fra = 'productID'

retailers = [1]#,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
results = pd.DataFrame(columns=['productID', 'variance', 'error'], index=retailers)
lags = [1, 2, 7, 14, 21]
for ret in retailers:
    a = d[generer_fra].value_counts()
    dp = d[d[generer_fra] == a.index[ret]]
    dp = dp.groupby(by='date').sum()
    dp = dp.resample('W').agg({'quantity': 'sum', 'turnover': 'sum', 'discount': 'sum'})
    dp = dp.dropna(axis=0, how='any')
    ts = dp['quantity']
    test_stationarity(ts, ret)
    for lag in lags:
        timemodeler(ts, ret, lag=lag)
    test = dp.groupby(by='date').sum()
    train = dp.groupby(by='date').sum()
    test = test.resample('W').agg({'quantity': 'sum', 'turnover': 'sum', 'discount': 'sum'})
    train = train.resample('W').agg({'quantity': 'sum', 'turnover': 'sum', 'discount': 'sum'})
    test = test.dropna(axis=0, how='any')
    train = train.dropna(axis=0, how='any')
    # test = test.iloc[::-1]
    # test = test.head(int(round(test.size * 0.1)))
    # train = train.head(int(round(train.size * 0.9)))

    # newmodel = linearmodeler(train)
    # columns = train.columns.tolist()
    # columns = [c for c in columns if c not in ['date', 'size', 'SupplierItemgroupName', 'styleNumber', 'colorname',
    #                                            'isNOS', 'styleNumber', 'description']]
    # newpred = newmodel.predict(test[columns])
    # print("Predictions clean:", newpred)
    # # Compute error between our test predictions and the actual values
    # lin_mse = mean_squared_error(newpred, test['quantity'])
    # print("Computed error:", lin_mse)
    # print('Variance score: %.2f' % r2_score(test['quantity'], newpred))
    # results.iloc[ret-1].set_value('productID', a.index[ret])
    # results.iloc[ret-1].set_value('variance', r2_score(test['quantity'], newpred))
    # results.iloc[ret-1].set_value('error', lin_mse)
    # predictionrange = pd.date_range(test.index.values[-1], periods=1, freq='W')
    # futurepred = newmodel.predict(predictionrange)
print(results)

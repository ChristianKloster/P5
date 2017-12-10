import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataloader as dl
from sklearn.linear_model import LinearRegression
import sklearn.metrics #import mean_squared_error, r2_score
import targetprovider
import os
import Featurerizer as FO
import math
import linreg
from sklearn.metrics import mean_squared_error as mse

def compare_models(M1, M2):
    MO = M1.copy()
    MN = M2.copy()

    SSEOld = MO['SSE'].mean()
    RMSEOld = MO['RMS'].mean()
    MSEOld = MO['MSE'].mean()
    MaxErrorOld = MO['MaxError'].mean()
    MOdesc = MO.describe()

    SSENew = MN['SSE'].mean()
    RMSENew = MN['RMS'].mean()
    MSENew = MN['MSE'].mean()
    MaxErrorNew = MN['MaxError'].mean()
    MNdesc = MN.describe()

    SSEComp = SSEOld-SSENew
    RMSEComp = RMSEOld-RMSENew
    MSEComp = MSEOld-MSENew
    MaxErrorComp = MaxErrorOld-MaxErrorNew
    Compdesc = MOdesc-MNdesc

    print('Old Model vs. New Model')
    print()
    print('SSE average difference:')
    print('SSE {0} vs. {1}'.format(SSEOld, SSENew))
    if SSEComp > 0:
        print('SSE new {0} better'.format(abs(SSEComp)))
    elif SSEComp == 0:
        print('SSE both same')
    else:
        print('SSE new {0} worse'.format(abs(SSEComp)))

    print()
    print('RMSE average difference:')
    print('RMSE {0} vs. {1}'.format(RMSEOld, RMSENew))
    if RMSEComp > 0:
        print('RMSE new {0} better'.format(abs(RMSEComp)))
    elif RMSEComp == 0:
        print('RMSE both same')
    else:
        print('RMSE new {0} worse'.format(abs(RMSEComp)))

    print()
    print('MSE average difference:')
    print('MSE {0} vs. {1}'.format(MSEOld, MSENew))
    if MSEComp > 0:
        print('MSE new {0} better'.format(abs(MSEComp)))
    elif MSEComp == 0:
        print('MSE both same')
    else:
        print('MSE new {0} worse'.format(abs(MSEComp)))

    print()
    print('Max Error average difference:')
    print('Max Error {0} vs. {1}'.format(MaxErrorOld, MaxErrorNew))
    if MaxErrorComp > 0:
        print('Max Error new {0} better'.format(abs(MaxErrorComp)))
    elif MaxErrorComp == 0:
        print('Max Error both same')
    else:
        print('Max Error new {0} worse'.format(abs(MaxErrorComp)))

    print()
    print('Old.describe')
    print(MOdesc)

    print()
    print('New.describe')
    print(MNdesc)

    print()
    print('Results of Old.describe - New.describe, negative tal er større fejl')
    print(Compdesc)

def error_plotter(df):
    data = df.copy()

    SSEframe = data['SSE']
    RMSframe = data['RMS']
    MSEframe = data['MSE']
    MaxEframe = data['MaxError']

    plt.close()
    plt.figure()
    SSEframe.plot()
    plt.xlabel('Iteration')
    plt.ylabel('Fejl')
    plt.title('SSE')
    plt.show()

    plt.close()
    plt.figure()
    RMSframe.plot()
    plt.xlabel('Iteration')
    plt.ylabel('Fejl')
    plt.title('RMS')
    plt.show()

    plt.close()
    plt.figure()
    MSEframe.plot()
    plt.xlabel('Iteration')
    plt.ylabel('Fejl')
    plt.title('MSE')
    plt.show()


    plt.close()
    plt.figure()
    MaxEframe.plot()
    plt.xlabel('Iteration')
    plt.ylabel('Fejl')
    plt.title('Max Error')
    plt.show()

kloster_dir = r'C:\Users\Christian\Desktop\Min Git mappe\P5\CleanData\\'
patrick_dir = r'C:\Users\Patrick\PycharmProjects\untitled\\'
ng_dir = r'C:\P5GIT\\'
print('Loading data...')
MNN = pd.read_csv(patrick_dir + 'NNRollNan01ErrorFrame1.rpt', sep=';').dropna(how='any')
MNB = pd.read_csv(patrick_dir + 'RollingFastNB11ErrorFrame1.rpt', sep=';').dropna(how='any')
MTree = pd.read_csv(patrick_dir + 'RollingFastTree11ErrorFrame1.rpt', sep=';').dropna(how='any')
MTreeE = pd.read_csv(patrick_dir + 'RollingFastTreeEkstra11ErrorFrame1.rpt', sep=';').dropna(how='any')
MLinLasso = pd.read_csv(patrick_dir + 'RollingLin1ErrorFrame.rpt', sep=';').dropna(how='any')
MNaiveBase = pd.read_csv(patrick_dir + '1Naive.rpt', sep=';')


print('Baseline vs Lin Lasso')
compare_models(MNaiveBase, MLinLasso)
print()
print('Baseline vs NB')
compare_models(MNaiveBase, MNB)
print()
print('Baseline vs Tree')
compare_models(MNaiveBase, MTree)
print()
print('Baseline vs Tree ekstra')
compare_models(MNaiveBase, MTreeE)
print()
print('Baseline vs NN')
compare_models(MNaiveBase, MNN)
print()
print('Lin vs NN')
compare_models(MLinLasso, MNN)
print()
print('Tree vs NB')
compare_models(MTree, MNN)
print()
print('Ekstra Tree vs Tree')
compare_models(MTreeE, MTree)
print()
print('NB vs NN')
compare_models(MNB, MNN)
print()
print('Tree vs NN')
compare_models(MTree, MNN)


error_plotter(MNB)
error_plotter(MTree)
error_plotter(MLinLasso)
error_plotter(MNN)

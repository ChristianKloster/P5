import dataloader as dl
from retailerdata import retailerData as rd
import pandas as pd
import numpy as np
from Plotter import plc as plc
import datetime
import matplotlib.pyplot as plt

#Generere en liste af transactions fra suppleret retailer og kigger på en defineret top % af dem, de grupperes efter generer_fra
def plotplc(retailer = 'alle', generer_fra = 'description', percent_of_interest = 20):
    # rddf = rd('C:/Users/SM-Baerbar/Documents/GitHub/P5/Retailers_w_coords.rpt')
    # d_danish = rddf.get_retailers_from_country('Denmark')
    # d_region = rddf.get_retailers_from_region('Sjælland')

    d = dl.load_sales_file('C:/Users/Patrick/PycharmProjects/untitled/CleanData/CleanedData.rpt')
    d = d.dropna(axis=0, how='any')
    d = d[d.isNOS != 1]
    #d = d[d.date.dt.day <=7]
    #d = d[d.retailerID.isin(d_danish.id)] #Kun danske butikke
    # d = d[d.retailerID.isin(d_region.id)] #Kun butikker i bestemt region
    #d = d[d.SupplierItemgroupName == 'MEN - JEANS']
    if retailer != 'alle':
        d = d[d.retailerID == retailer]

    #Plotter for alle produkter i én graf
    plc(d, 'PLC/{0}/{1}/{2}'.format(retailer, generer_fra, 'all'))
    # Laver en sorteret liste sorteret efter mængde forekomster
    a = d[generer_fra].value_counts()
    # Summere og har med overPercentDesc mulighed for at give en procentvis grænse for hvor stor en del af salget
    # der skal plottes for. 0.01 plotter kun varer der udgør 1% af total salg

    a = a.head(int(round(a.size * (percent_of_interest / 100))))
    for x in a:
        tilPLC = d[d[generer_fra] == a[a == x].index[0]]
        plc(tilPLC, 'PLC/{0}/{1}/{2}'.format(retailer, generer_fra, tilPLC['styleNumber'].iloc[0]))

    #Tager et productID eller et stylenumber og udprinter information om varen
def productreturn(id, stylenumber = 1):
    d = dl.load_sales_file('C:/Users/Patrick/PycharmProjects/untitled/CleanData/CleanedData.rpt')
    if stylenumber:
        d = d[d.styleNumber == id]
        print(d['colorname'].value_counts())
    else:
        d = d[d.productID == id]
        print(d[d.colorname].iloc[0])
    print(d.SupplierItemgroupName.iloc[0])
    print(d.description.iloc[0])

def featureplc(d, id):
    if type(id) is int:
        df = d[d.productID == id]
        df = d[d.styleNumber == df['styleNumber'].iloc[0]]
    else:
        df = d[d.styleNumber == id]
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
    return slope

def featurealder(d, id):
    if type(id) is int:
        df = d[d.productID == id]
    else:
        df = d[d.styleNumber == id]
    firstdate = df['date'].iloc[0]
    lastdate = df['date'].iloc[-1]
    current = pd.to_datetime('today')
    lifetime = lastdate - firstdate
    currenttime = current - firstdate
    return(lifetime.days, currenttime.days)

df = dl.load_sales_file('C:/Users/Patrick/PycharmProjects/untitled/CleanData/CleanedData.rpt')



# print(featurealder(df, 10721))
# print(featurealder(df, 'Z99319B'))
slope = featureplc(df, 'E02100A')
print(slope)
slope = slope.resample('W').sum()
print(slope)
plt.figure()
slope.plot()
plt.show()

# plotplc(3, generer_fra = 'description', percent_of_interest=20)
# plotplc(42, generer_fra = 'description', percent_of_interest=20)
# plotplc('alle', generer_fra='styleNumber', percent_of_interest=10)

# print(productreturn('F14307084', 1))
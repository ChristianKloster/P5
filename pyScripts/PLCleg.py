import dataloader as dl
from retailerdata import retailerData as rd
import pandas as pd
import numpy as np
from Plotter import plc as plc
import datetime
import matplotlib.pyplot as plt

#Generere en liste af transactions fra suppleret retailer og kigger på en defineret top % af dem, de grupperes efter generer_fra
def plotplc(retailer = 'alle', generer_fra = 'description', percent_of_interest = 20):
    #rddf = rd('C:/Users/SMSpin/Documents/GitHub/P5/Retailers_w_coords.rpt')
    # d_danish = rddf.get_retailers_from_country('Denmark')
    #Nordjylland, Midtjylland, Sønderjylland, Fyn, Sjælland
    #d_region = rddf.get_retailers_from_region('Sjælland')

    d = dl.load_sales_file(r'C:\Users\SMSpin\Documents\GitHub\P5\CleanData\CleanedData_no_isnos.rpt')
    d = d.dropna(axis=0, how='any')
    #d = d[d.isNOS != 1]
    #d = d[d.date.dt.day <=7]
    #d = d[d.retailerID.isin(d_danish.id)] #Kun danske butikke
    #d = d[d.retailerID.isin(d_region.id)] #Kun butikker i bestemt region
    #d = d[d.SupplierItemgroupName == 'MEN - JEANS']
    if retailer != 'alle':
        d = d[d.retailerID == retailer]

    #Plotter for alle produkter i én graf
    plc(d, 'PLC/{0}/{1}/{2}'.format(retailer, generer_fra, 'all'), ignore_returns = 0)
    # Laver en sorteret liste sorteret efter mængde forekomster
    a = d[generer_fra].value_counts()
    # Summere og har med overPercentDesc mulighed for at give en procentvis grænse for hvor stor en del af salget
    # der skal plottes for. 0.01 plotter kun varer der udgør 1% af total salg

    #a = a.head(int(round(a.size * (percent_of_interest / 100))))
    #for x in a:
    #    tilPLC = d[d[generer_fra] == a[a == x].index[0]]
    #    plc(tilPLC, 'PLC/{0}/{1}/{2}'.format(retailer, generer_fra, tilPLC['styleNumber'].iloc[0]))

    #Tager et productID eller et stylenumber og udprinter information om varen
def productreturn(d, id):
    if id in d.productID:
        d = d[d.productID == id]
        print(d[d.colorname].iloc[0])
    else:
        d = d[d.styleNumber == id]
        print(d['colorname'].value_counts())
    print(d.SupplierItemgroupName.iloc[0])
    print(d.description.iloc[0])
    print(d['retailerID'].value_counts())

def featureplcBD(d, id):
    #Relativt store udslag i hældningen, ikke nær så præcis som CD metoden, kan produceres for dags dato
    if id in d.productID:
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
    slope = pd.DataFrame(slope, columns=['slope'])
    return slope

def featureplcCD(d, id):
    #Denne giver mindre udslag i hældningerne og burde give et mere korrekt udtryk, men den kan ikke produceres for dags dato
    if id in d.productID:
        df = d[d.productID == id]
        df = d[d.styleNumber == df['styleNumber'].iloc[0]]
    else:
        df = d[d.styleNumber == id]
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
    slope = pd.DataFrame(slope, columns=['slope'])
    return slope

def featurealder(d, id):
    if id in d.productID:
        df = d[d.productID == id]
    else:
        df = d[d.styleNumber == id]
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

def featureacceleration(d, id):
    if id in d.productID:
        df = d[d.productID == id]
    else:
        df = d[d.styleNumber == id]
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
    acceleration = tempaccel.reindex(slope.index)#pd.DataFrame(slope, columns=['acceleration'])
    for x in tempaccel.index:
        acceleration.iloc[x] = tempaccel.iloc[x]
    acceleration = acceleration.fillna(value=0)
    return acceleration


plotplc('alle', percent_of_interest=100)
# df = df[df.retailerID == 4]


# print(featurealder(df, 10721))
# print(featurealder(df, 'Z99319B'))
# print(featurealder(df, 'E02100A'))

# slopeBD = featureplcBD(df, 'E02100A')
# slopeCD = featureplcCD(df, 'E02100A')
#print(featureacceleration(df, 'E02100A'))
#print(featureacceleration(df, 'Z99319B'))
#print(featureacceleration(df, 'F00001451'))
#print(featureacceleration(df, 'F00001460'))
# print(slopeBD)
# print(slopeCD)
# print(slopeBD-slopeCD)
# plt.figure()
# slopeBD.plot()
# plt.show()
# plt.figure()
# slopeCD.plot()
# plt.show()
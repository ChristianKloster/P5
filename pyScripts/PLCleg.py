import dataloader as dl
from retailerdata import retailerData as rd
import pandas as pd
from Plotter import plc as plc
import matplotlib.pyplot as plt

#Generere en liste af transactions fra suppleret retailer og kigger på en defineret top % af dem, de grupperes efter generer_fra
def plotplc(retailer = 'alle', generer_fra = 'description', percent_of_interest = 20):
    rddf = rd('C:/Users/SM-Baerbar/Documents/GitHub/P5/Retailers_w_coords.rpt')
    d_danish = rddf.get_retailers_from_country('Denmark')
    d_region = rddf.get_retailers_from_region('Sjælland')

    d = dl.load_sales_files(files)
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
    d = dl.load_sales_files(files)
    if stylenumber:
        d = d[d.styleNumber == id]
        print(d['colorname'].value_counts())
    else:
        d = d[d.productID == id]
        print(d[d.colorname].iloc[0])
    print(d.SupplierItemgroupName.iloc[0])
    print(d.description.iloc[0])

directory = 'C:/Users/SM-Baerbar/Documents/GitHub/P5/GOFACT_DATA/Sales_20'
files = ['1606', '1607', '1608', '1609','1610', '1611', '1612',
         '1701', '1702', '1703', '1704', '1705', '1706' , '1707', '1708', '1709']
end = '.rpt'

for x in range(0,len(files)):
	files[x] = directory + files[x] + end

# plotplc(3, generer_fra = 'description', percent_of_interest=20)
# plotplc(42, generer_fra = 'description', percent_of_interest=20)
plotplc('alle', percent_of_interest=100)

# print(productreturn('F14307084', 1))
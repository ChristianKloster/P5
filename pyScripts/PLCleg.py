import DataLoad as dl
import pandas as pd
from Plotter import plc as plc
import matplotlib.pyplot as plt

directory = 'C:/Users/Patrick/PycharmProjects/untitled/AAU/Sales_20'
files = ['1606', '1607', '1608', '1609','1610', '1611', '1612',
         '1701', '1702', '1703', '1704', '1705', '1706' , '1707', '1708', '1709']
end = '.rpt'

for x in range(0,len(files)):
	files[x] = directory + files[x] + end

d = dl.loadSalesFiles(files)
d = d.dropna(axis=0, how='any')
d = d.query("isNOS != 1")

genererFra = 'description'
a = d[genererFra].value_counts()
total = a.sum()
overPercentDesc = 0.01
overPercentStyle = 0.002
largerthan = total*overPercentDesc
toStyle = a > int(round(largerthan))
a= a[toStyle]
for x in a:
    forPLC = d[d[genererFra] ==  a[a == x].index[0]]
    # ignorer det der er udkommenteret her, det er noget bavl der kunne ende ud i en ikke naiv plotter
    # total = dDescriptions['quantity'].sum()
    # forPLC = dDescriptions.groupby(by='description').sum(axis='quantity', numeric_only=True)
    # toPLC = forPLC['quantity'] > total*overPercentStyle
    # forPLC = forPLC[toPLC]
    plc(forPLC, 'Gen;{0} Style;{2}'.format(genererFra, forPLC[genererFra].iloc[0],
                                             forPLC['styleNumber'].iloc[0]), 'place', 'holder')
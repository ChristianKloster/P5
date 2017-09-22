import pandas as pandas
import matplotlib.pyplot as plt
import Meta as mt


class Plotter:

    def plc(dataframe, name):
        #Bliver ordnet før denne metode, lige nu er det her til at lave test data
        dataframe = dataframe.dropna(axis=0, how='any')
        dataframe = dataframe.query("retailerId == 10")
        dataframe = dataframe.query("productId == 10024")
        #Parametre til metadata
        firstdate = dataframe['date'].iloc[0]
        lastdate = dataframe['date'].iloc[dataframe.ndim]
        retailer = dataframe['retailerId'].iloc[1]
        product = dataframe['productId'].iloc[1]
        #Sortering og plot
        dataframe = dataframe.groupby(by='date').sum(axis='Quantity', numeric_only=True)
        dataframe.plot(y='Quantity')
        #Gemmer lige nu til png fordi vi gerne vil have metadata, men pdf er bedre til brug i rapporten
        plt.savefig('%s.png' %name)
        #Viser plottet i en popup fra PyCharm
        plt.show()
        #Kalder metadata metoden og tilføjer info
        mt.addMetaData('%s.png' %name, { 'from':firstdate, 'to':lastdate,
                                         'retailer':retailer, 'product':product})
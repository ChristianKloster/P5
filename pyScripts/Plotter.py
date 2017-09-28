import matplotlib.pyplot as plt
import Meta as mt

def plc(dataframe, name, ret, prod):

        #Bliver ordnet før denne metode, lige nu er det her til at lave test data
        dataframe = dataframe.query("retailerID == @ret")
        dataframe = dataframe.query("productID == @prod")
        #Parametre til metadata
        # firstdate = dataframe['date'].iloc[0]
        # lastdate = dataframe['date'].iloc[-1]
        # retailer = str(dataframe['retailerID'].iloc[0])
        # product = str(dataframe['productID'].iloc[0])
        #Sortering og plot
        dataframe = dataframe.groupby(by='date').sum(axis='quantity', numeric_only=True)
        dataframe.plot(y='quantity', style='.')
        #Gemmer lige nu til png fordi vi gerne vil have metadata, men pdf er bedre til brug i rapporten
        plt.savefig('%s.png' %name)
        #Kalder metadata metoden og tilføjer info
        # mt.addMetaData('%s.png' %name, {'from':firstdate, 'to':lastdate,
        #                                  'retailer':retailer, 'product':product})

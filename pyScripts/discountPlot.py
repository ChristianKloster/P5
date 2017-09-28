from DataLoading import loadAllSales
import matplotlib.pyplot as plt

def plc(dataframe, name, ret, prod):

        dataframe = dataframe[dataframe.retailerID == ret]
        dataframe = dataframe[dataframe.productID == prod]
        dataframe = dataframe[dataframe.quantity >= 0]

        dataframe['discountP'] = (- dataframe['discount'] ) / (dataframe['turnover'] - dataframe['discount'])
        dataframe = dataframe.groupby(by='date').mean()
        dataframe.plot(y='discountP', style='.')

        plt.savefig('%s.png' %name)

# dp.plc(df,'test', 2, 64563)
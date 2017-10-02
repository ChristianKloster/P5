from DataLoading import loadAllSales
import matplotlib.pyplot as plt

def plc(dataframe, name, ret, prod):

        dataframe = dataframe[dataframe.retailerID == ret]
        dataframe = dataframe[dataframe.productID == prod]
        # ignore returns
        dataframe = dataframe[dataframe.quantity >= 0]

        # calculate discount as percentage
        dataframe['discountP'] = (- dataframe['discount'] ) / (dataframe['turnover'] - dataframe['discount']) * 100 

        # find mean percentage by date
        dataframe = dataframe.groupby(by='date').mean()
        dataframe.plot(y='discountP', style='.')

        plt.savefig('%s.png' %name)

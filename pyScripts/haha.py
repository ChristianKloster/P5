import linreg
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('featurized_ch1_new.csv', sep = ';', encoding = 'utf-8')



y = 'target_prod_agg_sun'
x = 'target_prod_rolling_7'

print(data[x].describe())
print(data[y].describe())

print(data.head(10))



data['diff'] = abs(data[x] - data[y])
df = data[data.productID == 10715]
df = df[df.retailerID == 7]
print(df[['date', 'quantity', x,y,'diff']])


plt.Figure()
plt.plot(data[x], data[y], 'b.')
plt.ylabel(y)
plt.xlabel(x)
plt.savefig(x +'_vs_'+ y + '.png')
plt.close()

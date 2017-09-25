import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

raw_data ={
	'black':    [1,2,3,4,5,6],
	'white':    [2,4,6,8,10,12],
	'gray':     [3,6,9,12,15,18],
	'blue':     [4,8,12,16,20,24],
	'red':      [5,10,15,20,25,30],
	'yellow':   [6,12,18,24,30,36],
	'purple':   [7,14,21,28,35,42],
}

def histogram(inputdata, styleName):
	df = pd.DataFrame(OrderedDict(inputdata),index=pd.Index(['size1', 'size2', 'size3', 'size4', 'size5', 'size6'], name=styleName))
	df.plot(kind='bar', color=(inputdata.keys()), edgecolor ='black')

	plt.show()

histogram(raw_data,'StyleName')

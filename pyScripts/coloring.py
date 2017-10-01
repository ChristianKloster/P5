import pandas as pd
import numpy as np

colorFrame = pd.read_csv('colors.csv', names=['color','amount'])

myColors = ['black','white','blue', 'red','green','grey','brown','yellow','rose', 'beige']
cs = colorFrame['color']
# make all lower case and remove whitepace before and after string
cs = cs.str.lower().str.strip()

colorDict = {}

for c in myColors:
	# get alle colorNames containing c form myColors
	s = cs[cs.str.contains(c)]
	# put list in dictionary
	colorDict[c] = s
	# remove colors from series
	cs = cs.drop(s.keys())

print(cs) # de farver der er tilbage

# TO DO:
# Make method that returns dict of color key and list of colernames:
# e.g.: {'black':['black', 'blacker black', 'slightly less black but still very black', 'blackish',...],...}

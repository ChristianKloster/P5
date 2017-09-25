import pandas as pd
import numpy as np
import re

colorFrame = pd.read_csv('colors.csv', names=['color','amount'])

myColors = ['black','white','blue', 'red','green','grey','brown','yellow','rose', 'beige']
cs = colorFrame['color']
cs = cs.str.lower().str.strip()

colorDict = {}

for c in myColors:
	s = cs[cs.str.contains(c)]
	colorDict[c] = s
	cs = cs.drop(s.keys())

print(cs) # de farver der er tilbage

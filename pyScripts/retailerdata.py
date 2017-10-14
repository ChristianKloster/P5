import pandas as pd

class retailerData:
	def __init__(self, path):
		self.df = pd.read_csv(path ,sep=';', comment='(',encoding='latin-1')

	def get_dataframe(self):
		return self.df.copy()

	# returns int chainid
	def get_chainnumber_from_retailerid(self, id):
		s = self.df[self.df.id == id]
		return s['chainid'].get_value(0)

	# takes string (case sensitive) returns dataframe
	def get_retailers_from_country(self, country):
		return self.df[self.df.country == country]

	# returns dataframe
	def get_retailers_from_chainid(self, id):
		return self.df[self.df.chainid == id]


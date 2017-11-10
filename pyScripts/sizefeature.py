
class SizeFeature:
	def __init__(self):

# disse bør nok laves til nogle tabeller vi kan præsentere:

		self.mapping_ch1 = {'XS' : 1,
							'S' : 2,
							'S/M' : 3,
							'M' : 4,
							'M/L' : 5,
							'L' : 6,
							'L/XL' : 7,
							'XL' : 8,
							'XXL' : 10,
							'40' : 1 ,
							'42' : 2,
							'44' : 3,
							'46' : 4,
							'48' : 5,
							'50' : 6,
							'52' : 7,
							'54' : 8,
							'56' : 9,
							'105' : 2,
							'115' : 4,
							'125' : 6,
							'ONE SIZE' : 0,
							'Unknown' : 9999}

		self.mapping_ch2_w = {'XXS' : 1,
							'XS' : 2,
							'XS/S' : 3, 
							'S' : 4,
							'S/M' : 5,
							'M' : 6,
							'M/L' : 7,
							'L' : 8,
							'L/XL' : 9,
							'XL' : 10,
							'XXL' : 11,
							'24\"' : 1,
							'25\"' : 2,
							'26\"' : 3,
							'27\"' : 4,
							'28\"' : 5,
							'29\"' : 6,
							'30\"' : 8,
							'31\"' : 9,
							'32\"' : 10,
							'33\"' : 11,
							'34\"' : 12,
							'36\"' : 13,
							'37-40' : 7,
							'41-46' : 9,
							'36' : 6,
							'37' : 6,
							'38' : 6,
							'39' : 7,
							'40' : 8,
							'41' : 8,
							'42' : 8,
							'43' : 9,
							'44' : 10,
							'45' : 10,
							'46' : 10,
							'48' : 11,
							'50' : 12,
							'52' : 12,
							'54' : 13,
							'56' : 13,
							'85' : 1,
							'90' : 2,
							'95' : 4,
							'100' : 6,
							'105' : 9,
							'*** ONE SIZE ***' : 0,
							'ONE SIZE' : 0,
							'ONE' : 0,
							'Unknown' : 9999 }

		self.mapping_ch2_m = {'XXS' : 1,
							'XS' : 2,
							'XS/S' : 3, 
							'S' : 4,
							'S/M' : 5,
							'M' : 6,
							'M/L' : 7,
							'L' : 8,
							'L/XL' : 9,
							'XL' : 10,
							'XXL' : 11,
							'24\"' : 1,
							'25\"' : 2,
							'26\"' : 3,
							'27\"' : 4,
							'28\"' : 6,
							'29\"' : 8,
							'30\"' : 9,
							'31\"' : 10,
							'32\"' : 11,
							'33\"' : 11,
							'34\"' : 12,
							'36\"' : 13,
							'37-40' : 1,
							'41-46' : 2,
							'36' : 1,
							'37' : 1,
							'38' : 1,
							'39' : 1,
							'40' : 1,
							'41' : 1,
							'42' : 1,
							'43' : 1,
							'44' : 2,
							'45' : 3,
							'46' : 4,
							'48' : 6,
							'50' : 8,
							'52' : 10,
							'54' : 11,
							'56' : 12,
							'85' : 1,
							'90' : 4,
							'95' : 8,
							'100' : 10,
							'105' : 11,
							'*** ONE SIZE ***' : 0,
							'ONE SIZE' : 0,
							'ONE' : 0,
							'Unknown' : 9999}

		self.mapping_ch3  = {'XS' : 1,
							'XS/S': 2,
							'S' : 3,
							'S/M' : 4,   
							'M' : 5,
							'M/L' :6,  
							'L' :7,
							'ONE SIZE' : 0,
							'Unknown' : 9999}

	def get_size_feature(self, size, chainid, male = False):
		if chainid == 1:
			return self.mapping_ch1[size]
		elif chainid == 2:
			if male:
				return self.mapping_ch2_m[size]
			else:
				return self.mapping_ch2_w[size]
		elif chainid == 3:
			return self.mapping_ch3[size]
		else: 
			return 8888 

# requires that chinid and ismale is present
def make_sizefeature_col(df):
	sf = SizeFeature()

	data = df.copy()
	data['size_scale'] = tuple(map(lambda size, chainid, ismale: sf.get_size_feature(size=size, chainid = chainid, male = ismale), data['size'], data['chainid'], date['ismale']))
	return data

# zizzi
# Vores styles er altid udviklet efter europæiske måle standarder.
# https://www.zizzi.dk/hjaelp/guides/stoerrelsesguide

# samsoe women : http://www.sizeguide.net/womens-clothing-sizes-international-conversion-chart.html
# http://www.samsoe.com/da/support/size-guide/mens-jeans.html
 







  
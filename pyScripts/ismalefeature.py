



mens_itemgroup = ['MEN - ACCESSORIES',
				'MEN - BASIC - DON\'T USE',
				'MEN - BELTS/SCARF/TIE',
				'MEN - BLAZER',
				'MEN - CARDIGAN',
				'MEN - JACKETS',
				'MEN - JEANS',
				'MEN - KNIT',
				'MEN - PANTS',
				'MEN - POLO',
				'MEN - SHIRTS',
				'MEN - SHOES', 
				'MEN - SHORTS',
				'MEN - SWEAT',               
				'MEN - T-SHIRTS',            
				'MEN - UNDERWEAR/SOCKS']    

def is_male(itemgroupname):
	return 1 if itemgroupname in mens_itemgroup else 0

print(is_male('MEN - SHORTS'), is_male('fisk'))
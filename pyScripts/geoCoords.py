import requests as rq
from pprint import pprint

apiservice = 'https://maps.googleapis.com/maps/api/geocode/json'
apikey = 'AIzaSyDlNh8UqVinENZuX_Bm2O8Bp6zrOToHQ3o'

def main():
	print(getCoords('aalborg', 'denmark'))

def getCoords(city, country):
	args = 'address='+ city + ',+' + country
	s = makeRequestString(args)
	response = rq.get(s)
	data = response.json()
	return data['results'][0]['geometry']['location']

def makeRequestString(args):
	return apiservice + '?' + args + '&key=' + apikey

if __name__ == '__main__':
	main()
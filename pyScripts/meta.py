# Use PIL (pillow) to save some image metadata
from PIL import Image
from PIL import PngImagePlugin

# inserts metadata e.g.: {"version":"1.0", "author":"nicolai"}
def addMetaData(path, data):
	img = Image.open(path)
	meta = PngImagePlugin.PngInfo()

	for x in data:
		meta.add_text(x, data[x])
		img.save(path, "png", pnginfo=meta)

def getMetaData(path):
	img= Image.open(path)
	return img.info

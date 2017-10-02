# Script for viewing the metadata af a png file
# type HiMeta {path}
# or just HiMeta, and the script will promt for a path

import sys
import meta as mt

l = len(sys.argv)


# no command line arguments
# 2nd arg = path of png file 
if l == 1:
	path = input("Enter path here: ")
elif l == 2:
	path = sys.argv[1]
else:
	print("Too many arguments! Can't handle the pressure! Shutting down!")
	sys.exit()

print(mt.getMetaData(path))

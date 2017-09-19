import sys
import meta as mt

l = len(sys.argv)

if l == 1:
	path = input("Enter path here: ")
elif l == 2:
	path = sys.argv[1]
else:
	print("Too many arguments! Can't handle the pressure! Shutting down!")
	sys.exit()

print(mt.getMetaData(path))

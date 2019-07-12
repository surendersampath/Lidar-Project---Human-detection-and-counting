try:
	from PyQt5 import *
	print('Using PyQt5')
except ImportError:
	try:
		from PyQt4 import *
		print('Using PyQt4')
	except ImportError:
		print('Neither PyQt5 nor PyQt4 found, please install at least one.')
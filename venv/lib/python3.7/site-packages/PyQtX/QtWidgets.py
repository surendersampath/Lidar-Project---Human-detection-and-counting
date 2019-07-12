try:
	from PyQt5.QtWidgets import *
except ImportError:
	from PyQt4.QtGui import *
	from PyQt4.QtGui import QFileDialog as OldFileDialog


	class QFileDialog(OldFileDialog):
		@staticmethod
		def getOpenFileName(
				parent=None, 
				caption='', 
				directory='', 
				filter='', 
				selectedFilter='',
				options=QFileDialog.Options()
			):
			return OldFileDialog.getOpenFileNameAndFilter(
				parent, 
				caption, 
				directory, 
				filter, 
				selectedFilter,
				options
			)

		@staticmethod
		def getOpenFileNames(
				parent=None, 
				caption='', 
				directory='', 
				filter='', 
				selectedFilter='',
				options=QFileDialog.Options()
			):
			return OldFileDialog.getOpenFileNamesAndFilter(
				parent, 
				caption, 
				directory, 
				filter, 
				selectedFilter,
				options
			)
			

		@staticmethod
		def getSaveFileName(
				parent=None, 
				caption='', 
				directory='', 
				filter='', 
				selectedFilter='',
				options=QFileDialog.Options()
			):
			return OldFileDialog.getSaveFileNameAndFilter(
				parent, 
				caption, 
				directory, 
				filter, 
				selectedFilter,
				options
			)
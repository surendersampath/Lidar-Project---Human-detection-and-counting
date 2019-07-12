from abc import ABCMeta
from abc import abstractproperty
from collections import namedtuple

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QModelIndex
from PyQt5.QtCore import QAbstractItemModel


#    @property
#    def row(self):
#        if self._parent:
#            return self._parent.children.index(self)
#        return 0


_ColumnDef = namedtuple("ColumnDef", ["displayName", "attributeName", "formatter"])
def ColumnDef(displayName, attributeName, formatter=str):
    return _ColumnDef(displayName, attributeName, formatter)


class TreeModel(QAbstractItemModel):
    """ adapter from Item to QAbstractItemModel interface """
    def __init__(self, root, columns, parent=None):
        super(TreeModel, self).__init__(parent)
        self._root = root
        self._columns = columns

    def columnCount(self, parent):
        return len(self._columns)

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._columns[section].displayName
        return None

    def data(self, index, role):
        if not index.isValid():
            return None

        if role != Qt.DisplayRole:
            return None

        item = index.internalPointer()
        coldef = self._columns[index.column()]
        return coldef.formatter(getattr(item, coldef.attributeName))

    def index(self, row, column=0, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parentItem = self._root
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.children[row]
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent

        if parentItem == self._root:
            return QModelIndex()

        return self.createIndex(parentItem.row, 0, parentItem)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self._root
        else:
            parentItem = parent.internalPointer()

        return len(parentItem.children)

    def get_root_index(self):
        pass

    def treeChanged(self):
        self.layoutChanged.emit()


# TODO: fix bug of bordering zero-length item

import os
import imp
import functools
import binascii
from abc import ABCMeta
from abc import abstractproperty

from funcy import cached_property
from hexview import QT_COLORS
from hexview import HexViewWidget
from hexview import make_color_icon

from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMenu
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QInputDialog

from vstruct import VStruct
from vstruct import VArray
from vstruct.primitives import v_prim
from vstruct.primitives import v_number
from vstruct.primitives import v_bytes
from vstruct.primitives import v_uint8
from vstruct.primitives import v_uint16
from vstruct.primitives import v_uint32

# need to import this so that defs can import vstructui and access the class
from .vstruct_parser import VstructParserSet
from .vstruct_parser import BasicVstructParserSet

from .common import h
from .common import LoggingObject
from .tree import TreeModel
from .tree import ColumnDef
from .vstruct_parser import ComposedParser
from .vstruct_parser import VstructInstance

from .vstructui_auto import Ui_Form as VstructViewBase


defspath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "defs")
def get_parsers(defspath=defspath):
    parsers = ComposedParser()
    for filename in os.listdir(defspath):
        if not filename.endswith(".py"):
            continue
        deffilepath = os.path.join(defspath, filename)
        mod = imp.load_source("vstruct_parser", deffilepath)
        if not hasattr(mod, "vsEntryVstructParser"):
            continue
        parser = mod.vsEntryVstructParser()
        parsers.add_parser(parser)
    return parsers


class Item(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def parent(self):
        raise NotImplementedError()

    @abstractproperty
    def children(self):
        raise NotImplementedError()

    @abstractproperty
    def name(self):
        raise NotImplementedError()

    @abstractproperty
    def type(self):
        raise NotImplementedError()

    @abstractproperty
    def data(self):
        raise NotImplementedError()

    @abstractproperty
    def offset(self):
        raise NotImplementedError()

    @abstractproperty
    def length(self):
        raise NotImplementedError()

    @abstractproperty
    def end(self):
        raise NotImplementedError()


class VstructItem(Item):
    def __init__(self, instance, parent=None):
        super(VstructItem, self).__init__()
        self._instance = instance
        self._parent = parent

    @property
    def name(self):
        return self._instance.name

    @name.setter
    def name(self, value):
        self._instance.name = value

    @property
    def offset(self):
        return self._instance.offset

    @property
    def parent(self):
        return self._parent

    def __repr__(self):
        return "VstructItem(name: {:s}, type: {:s}, start: {:s}, length: {:s}, end: {:s})".format(
            self.name,
            self.type,
            h(self.offset),
            h(self.length),
            h(self.end),
        )

    @cached_property
    def children(self):
        ret = []
        if isinstance(self._instance.instance, VStruct):
            off = self.offset
            # TODO: don't reach
            for fname in self._instance.instance._vs_fields:
                x = self._instance.instance._vs_values.get(fname)
                ret.append(VstructItem(VstructInstance(off, x, fname), self))
                off += len(x)
        return ret

    @property
    def type(self):
        return self._instance.instance.__class__.__name__

    @property
    def data(self):
        i = self._instance.instance
        if isinstance(i, VStruct):
            return ""
        elif isinstance(i, v_number):
            if i.vsGetEnum() is not None:
                return str(i)
            else:
                return h(i.vsGetValue())
        elif isinstance(i, v_bytes):
            return binascii.b2a_hex(i.vsGetValue())
        elif isinstance(i, v_prim):
            return i.vsGetValue()
        else:
            return ""

    @property
    def length(self):
        return len(self._instance.instance)

    @property
    def end(self):
        return self.offset + self.length

    @property
    def row(self):
        if self._parent:
            return self._parent.children.index(self)
        return 0

    def __cmp__(self, other):
        if self.offset == other.offset:
            return self.length - other.length
        return self.offset - other.offset

    def __lt__(self, other):
        return self.__cmp__(other) < 0


class VstructRootItem(Item):
    def __init__(self, instances):
        super(VstructRootItem, self).__init__()
        self._items = sorted([VstructItem(i, self) for i in instances])

    @property
    def parent(self):
        return None

    def __repr__(self):
        return "VstructRootItem()"

    @property
    def children(self):
        return self._items

    def add_item(self, instance):
        # be sure to call the TreeModel.treeChanged() method
        self._items.append(VstructItem(instance, self))
        self._items = sorted(self._items)

    @property
    def name(self):
        return None

    @property
    def type(self):
        return None

    @property
    def data(self):
        return None

    @property
    def offset(self):
        return 0

    @property
    def length(self):
        return 0

    @property
    def end(self):
        return 0


class VstructHexViewWidget(HexViewWidget):
    # args: offset, parser_name
    parseRequested = pyqtSignal([int, str])

    def __init__(self, parsers, *args, **kwargs):
        super(VstructHexViewWidget, self).__init__(*args, **kwargs)
        self._parsers = parsers

    def get_context_menu(self, qpoint):
        menu = super(VstructHexViewWidget, self).get_context_menu(qpoint)

        sm = self.getSelectionModel()
        if sm.start != sm.end:
            return menu

        def add_action(menu, text, handler, icon=None):
            a = None
            if icon is None:
                a = QAction(text, self)
            else:
                a = QAction(icon, text, self)
            a.triggered.connect(handler)
            menu.addAction(a)

        parser_menu = menu.addMenu("Parse as...")
        for parser_name in sorted(self._parsers.parser_names):
            add_action(parser_menu, parser_name,
                       functools.partial(self.parseRequested.emit, sm.start, parser_name))

        return menu


class VstructViewWidget(QWidget, VstructViewBase, LoggingObject):
    def __init__(self, parsers, instances, buf, parent=None):
        """ items is a list of VstructItem """
        super(VstructViewWidget, self).__init__(parent)
        self.setupUi(self)

        self._buf = buf
        self._root_item = VstructRootItem(instances)
        self._model = TreeModel(
            self._root_item,
            [
                ColumnDef("Name", "name"),
                ColumnDef("Type", "type"),
                ColumnDef("Data", "data"),
                ColumnDef("Start", "offset", formatter=h),
                ColumnDef("Length", "length", formatter=h),
                ColumnDef("End", "end", formatter=h),
            ])
        self._parsers = parsers

        self._hv = VstructHexViewWidget(self._parsers, self._buf, self.splitter)
        self._hv.parseRequested.connect(self._handle_parse_requested)
        self.splitter.insertWidget(0, self._hv)

        tv = self.treeView
        tv.setModel(self._model)
        tv.header().setSectionResizeMode(QHeaderView.Interactive)

        # used for mouse click
        tv.clicked.connect(self._handle_item_clicked)
        # used for keyboard navigation
        tv.selectionModel().selectionChanged.connect(self._handle_item_selected)

        tv.setContextMenuPolicy(Qt.CustomContextMenu)
        tv.customContextMenuRequested.connect(self._handle_context_menu_requested)

        # used to track the current "selection" controlled by the tree view entries
        self._current_range = None  # type: Pair[int, int]

    def _clear_current_range(self):
        if self._current_range is None:
            return
        self._hv.getBorderModel().clear_region(*self._current_range)
        self._current_range = None

    def _handle_item_clicked(self, item_index):
        self._handle_item_activated(item_index)

    def _handle_item_selected(self, item_indices):
        # hint found here: http://stackoverflow.com/a/15214966/87207
        if not item_indices.indexes():
            self._clear_current_range()
        else:
            self._handle_item_activated(item_indices.indexes()[0])

    def _color_item(self, item, color=None):
        start = item.offset
        end = start + item.length
        # deselect any existing ranges, or else colors get confused
        self._hv._hsm.bselect(-1, -1)
        return self._hv.getColorModel().color_region(start, end, color)

    def _is_item_colored(self, item):
        start = item.offset
        end = start + item.length
        return self._hv.getColorModel().is_region_colored(start, end)

    def _clear_item(self, item):
        start = item.offset
        end = start + item.length
        return self._hv.getColorModel().clear_region(start, end)

    def _handle_item_activated(self, item_index):
        self._clear_current_range()
        item = item_index.internalPointer()
        start = item.offset
        end = start + item.length
        self._hv.getBorderModel().border_region(start, end, Qt.black)
        self._current_range = (start, end)
        self._hv.scrollTo(start)

    def _handle_context_menu_requested(self, qpoint):
        index = self.treeView.indexAt(qpoint)
        item = index.internalPointer()

        def add_action(menu, text, handler, icon=None):
            a = None
            if icon is None:
                a = QAction(text, self)
            else:
                a = QAction(icon, text, self)
            a.triggered.connect(handler)
            menu.addAction(a)

        menu = QMenu(self)

        action = None
        if self._is_item_colored(item):
            add_action(menu, "De-color item", lambda: self._handle_clear_color_item(item))
        else:
            add_action(menu, "Color item", lambda: self._handle_color_item(item))
            color_menu = menu.addMenu("Color item...")

            # need to escape the closure capture on the color loop variable below
            # hint from: http://stackoverflow.com/a/6035865/87207
            def make_color_item_handler(item, color):
                return lambda: self._handle_color_item(item, color=color)

            for color in QT_COLORS:
                add_action(color_menu, "{:s}".format(color.name),
                           make_color_item_handler(item, color.qcolor), make_color_icon(color.qcolor))

        add_action(menu, "Set name...", lambda: self._handle_set_name(item))

        menu.exec_(self.treeView.mapToGlobal(qpoint))

    def _handle_color_item(self, item, color=None):
        self._color_item(item, color=color)

    def _handle_clear_color_item(self, item):
        self._clear_item(item)

    def _handle_parse_requested(self, offset, parser_name):
        for vi in self._parsers.parse(parser_name, self._buf, offset, ""):
            self._root_item.add_item(vi)
        self._model.treeChanged()

    def _handle_set_name(self, item):
        name, ok = QInputDialog.getText(self, "Set name...", "Name:")
        if ok and name:
            item.name = name
            self._model.treeChanged()


_HEX_ALPHA_CHARS = set(list("abcdefABCDEF"))
def is_probably_hex(s):
    if s.startswith("0x"):
        return True

    for c in s:
        if c in _HEX_ALPHA_CHARS:
            return True

    return False


def main(*args):
    parsers = get_parsers()
    buf = ""
    structs = ()
    if len(args) == 0:
        b = []
        for i in range(0x100):
            b.append(i)
        buf = bytearray(b)

        class TestStruct(VStruct):
            def __init__(self):
                VStruct.__init__(self)
                self.a = v_uint8()
                self.b = v_uint16()
                self.c = v_uint32()
                self.d = v_uint8()
                self.e = VArray((v_uint32(), v_uint32(), v_uint32(), v_uint32()))

        t1 = TestStruct()
        t1.vsParse(buf, offset=0x0)

        t2 = TestStruct()
        t2.vsParse(buf, offset=0x40)
        structs = (VstructInstance(0x0, t1, "t1"), VstructInstance(0x40, t2, "t2"))
    else:
        # vstructui.py /path/to/binary/file "0x0:uint32:first dword" "0x4:uint_2:first word"
        structs = []
        args = list(args)  # we want a list that we can modify
        filename = args.pop(0)
        with open(filename, "rb") as f:
            buf = f.read()

        for d in args:
            if ":" not in d:
                raise RuntimeError("invalid structure declaration: {:s}".format(d))

            soffset, _, parser_name = d.partition(":")
            name = ""
            if ":" in parser_name:
                parser_name, _, name = parser_name.partition(":")
            offset = None
            if is_probably_hex(soffset):
                offset = int(soffset, 0x10)
            else:
                offset = int(soffset)

            structs.extend(parsers.parse(parser_name, buf, offset, name=name))

    app = QApplication(sys.argv)
    screen = VstructViewWidget(parsers, structs, buf)
    screen.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])

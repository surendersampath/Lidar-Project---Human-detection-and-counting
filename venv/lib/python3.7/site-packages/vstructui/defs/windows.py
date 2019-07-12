import struct
import datetime

import vstruct
from vstructui import BasicVstructParserSet


class FILETIME(vstruct.primitives.v_prim):
    _vs_builder = True
    def __init__(self):
        vstruct.primitives.v_prim.__init__(self)
        self._vs_length = 8
        self._vs_value = "\x00" * 8
        self._vs_fmt = "<Q"
        self._ts = datetime.datetime.min

    def vsParse(self, fbytes, offset=0):
        offend = offset + self._vs_length
        q = struct.unpack("<Q", fbytes[offset:offend])[0]
        self._ts = datetime.datetime.utcfromtimestamp(float(q) * 1e-7 - 11644473600 )
        return offend

    def vsEmit(self):
        raise NotImplementedError()

    def vsSetValue(self, guidstr):
        raise NotImplementedError()

    def vsGetValue(self):
        return self._ts

    def __repr__(self):
        return self._ts.isoformat("T") + "Z"


def vsEntryVstructParser():
    return BasicVstructParserSet((FILETIME,))

from vstruct.primitives import v_int8
from vstruct.primitives import v_uint8
from vstruct.primitives import v_int16
from vstruct.primitives import v_uint16
from vstruct.primitives import v_int32
from vstruct.primitives import v_uint32
from vstruct.primitives import v_int64
from vstruct.primitives import v_uint64
from vstruct.primitives import v_double
from vstruct.primitives import v_float
from vstruct.primitives import GUID
from vstruct.primitives import v_zstr
from vstruct.primitives import v_wstr

from vstructui import VstructParserSet


def vsEntryVstructParser():
    s = VstructParserSet()
    for t in (v_int8, v_uint8, v_int16, v_uint16, v_int32, v_uint32, v_int64,
              v_uint64, v_double, v_float, v_uint64, v_zstr, v_wstr):
        s.register_basic_parser(t.__name__.lstrip("v_"), t)
    s.register_basic_parser("GUID", GUID)
    return s


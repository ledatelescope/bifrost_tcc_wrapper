
from bifrost.libbifrost import _check, _get, BifrostObject
from bifrost.ndarray import asarray

import btcc_generated as _gen

class Btcc(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _gen.BTccCreate, _gen.BTccDestroy)
    def init(self, nbits_c_int, ntime_c_int, nchan_c_int, nstand_c_int, npol_c_int, decim_c_int):
        _check(_gen.BTccInit(self.obj, nbits_c_int, ntime_c_int, nchan_c_int,
                             nstand_c_int, npol_c_int, decim_c_int))

    def execute(self, in_BFarray, out_BFarray, dump_BFbool):
        _check(_gen.BTccExecute(self.obj, asarray(in_BFarray).as_BFarray(),
                                asarray(out_BFarray).as_BFarray(),
                                dump_BFbool))
        return out_BFarray

    def set_stream(self, stream_ptr_generic):
        _check(_gen.BTccSetStream(self.obj, stream_ptr_generic))

    def reset_state(self):
        _check(_gen.BTccResetState(self.obj))


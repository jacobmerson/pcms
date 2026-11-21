"""Process-wide Omega_h library; call shutdown() after all pcms objects are
released.
"""

import atexit
import gc

import PyOmega_h as omega_h

_library = None


def library():
    global _library
    if _library is None:
        _library = omega_h.OmegaHLibrary()
        atexit.register(shutdown)
    return _library


def shutdown():
    global _library
    gc.collect()
    _library = None
    gc.collect()

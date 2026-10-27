"""Process-wide Omega_h library (and thus Kokkos) lifetime.

Kokkos may be initialized once per process, and the library must outlive
every pcms/Omega_h object or Kokkos aborts at exit. Scripts and tests get the
library from here; call shutdown() after all pcms objects are released.
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

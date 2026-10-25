#ifndef PCMS_NIMROD_FFI_H
#define PCMS_NIMROD_FFI_H

// Foreign-function declarations for NIMROD's C-callable field evaluator.
//
// These map the BIND(C) entry points defined in NIMROD's
//   nimdevel/nimcore/eval_field.F90
// which are packaged in the `nimeval` shared library. PCMS calls the same
// symbols that nimpy's EvalNimrod (nimpy/eval_nimrod.py) wraps via ctypes, so
// evaluation results agree with nimpy by construction and PCMS does not
// reimplement NIMROD's bases, point location, or Fourier reconstruction.
//
// All Fortran BIND(C) arguments are passed by reference, i.e. as pointers here
// (mirroring nimpy, which passes every argument via ctypes.byref).
//
// NOTE: this interface is process-global and holds a single active dump: a
// second call to c_nimfield_init re-initializes over the first. The C++
// wrappers in nimrod.h guard this.

extern "C"
{

  // Read a dump file and initialize the requested fields for evaluation.
  //   dumpname:  NUL-terminated path to the NIMROD dump file.
  //   fieldlist: NUL-terminated list of single-character field ids to
  //              initialize (e.g. "bvnpf").
  // Requires nimrod.in to be present in the working directory.
  void c_nimfield_init(const char* dumpname, const char* fieldlist);

  // Evaluate a real (physical-space) field at a single (R,Z,Phi) point.
  //   rzp:     [3] query point in cylindrical coordinates.
  //   fname:   NUL-terminated single-character field id (e.g. "b").
  //   fval:    [nqty*nderiv] output; nderiv = 1,4,10 for dmode = 0,1,2.
  //            Component-major, derivative-minor (fval[iqty + ider*nqty]).
  //   nqty:    number of vector components for this field.
  //   xyguess: [2] in/out logical-coordinate guess used to seed point
  //            location; updated on success. Seed with (-1,-1) to search.
  //   ifail:   out; nonzero if the point is out of bounds.
  //   dmode:   derivative mode: 0 (value), 1 (+first), 2 (+second).
  //   eq:      equilibrium/perturbation selector (see eval_field.F90):
  //            1 eq, 0 pert, 2 eq+pert, 3 eq+n0, 4 pert-n0.
  void c_eval_field_dm(const double* rzp, const char* fname, double* fval,
                       const int* nqty, double* xyguess, int* ifail,
                       const int* dmode, const int* eq);

  // Evaluate a complex (per-Fourier-mode) field at a single (R,Z) point.
  //   rz:    [2] query point in the poloidal plane.
  //   fval:  [2*nqty*nmodes*nderiv] output, interleaved real/imag; layout per
  //          nimpy __eval_comp_field_on_point (real+imag pairs).
  //   Other arguments as c_eval_field_dm (no eq selector).
  void c_eval_comp_field_dm(const double* rz, const char* fname, double* fval,
                            const int* nqty, double* xyguess, int* ifail,
                            const int* dmode);

} // extern "C"

#endif // PCMS_NIMROD_FFI_H

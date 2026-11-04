#ifndef PCMS_FIELD_FUNCTION_SPACE_NIMEVAL_HPP
#define PCMS_FIELD_FUNCTION_SPACE_NIMEVAL_HPP

// Private (not installed): NIMROD's C-callable evaluator, for the NIMROD
// function space and its tests.

// BIND(C) entry points of NIMROD's nimeval library (nimcore/eval_field.F90),
// the ones nimpy's EvalNimrod wraps. Every argument is passed by reference.
extern "C" {
// Reads a dump and initializes the fields named by the single-character ids
// in fieldlist. Reads nimrod.in from, and writes scratch files to, the
// current directory.
void c_nimfield_init(const char* dumpname, const char* fieldlist);

// Evaluates field fname at the (R, Z, phi) point rzp into fval[nqty] (for
// dmode 0). xyguess seeds the point search and is updated; ifail is nonzero
// if the point is outside the domain. eq: 0 perturbation, 1 equilibrium,
// 2 both.
void c_eval_field_dm(const double* rzp, const char* fname, double* fval,
                     const int* nqty, double* xyguess, int* ifail,
                     const int* dmode, const int* eq);
}

#endif // PCMS_FIELD_FUNCTION_SPACE_NIMEVAL_HPP

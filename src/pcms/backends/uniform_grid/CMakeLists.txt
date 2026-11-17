# Uniform-grid backend (no external dependency). No public headers.

# layout, evaluators and spline interpolators: part of pcms::field
target_sources(
  pcms_field
  PRIVATE layout.cpp layout.hpp evaluator.hpp spline_interpolator.hpp
          spline_evaluator.hpp geometry.hpp linear_interpolant.hpp
          multidimarray.hpp)

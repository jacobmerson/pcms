# Track A: fields and evaluation in a single process.

add_subdirectory(hello-field)
add_subdirectory(point-clouds)
add_subdirectory(splines-eqdsk)

# these build Omega_h meshes
foreach(example mesh-fields moving-points coordinate-systems)
  if(PCMS_EXAMPLES_HAVE_OMEGA_H)
    add_subdirectory(${example})
  else()
    pcms_examples_skip("fields/${example}" "PCMS was built without Omega_h")
  endif()
endforeach()

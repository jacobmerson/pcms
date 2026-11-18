pcms_add_example(fields_splines_eqdsk main.cpp CORE_ONLY
                 ARGS "${CMAKE_CURRENT_SOURCE_DIR}/solovev.eqdsk")
pcms_add_python_example(main.py ARGS "${CMAKE_CURRENT_SOURCE_DIR}/solovev.eqdsk")

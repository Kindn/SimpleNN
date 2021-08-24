include_directories( ${CMAKE_SOURCE_DIR}/include
                   ${CMAKE_SOURCE_DIR}/util )

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix functions)

add_executable(test_functions test_functions.cpp)
target_link_libraries(test_functions functions layers)

add_executable(test_layers test_layers.cpp)
target_link_libraries(test_layers functions layers)

add_executable(test_io test_io.cpp)
target_link_libraries(test_io nets layers functions data )

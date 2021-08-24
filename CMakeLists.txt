cmake_minimum_required ( VERSION 2.8.5 )

project ( SimpleNN )

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
set (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O2")

add_subdirectory ( src )
#add_subdirectory ( util )
add_subdirectory ( test )
add_subdirectory( examples )


include_directories( ${CMAKE_SOURCE_DIR}/util
${CMAKE_SOURCE_DIR}/include )

file ( GLOB SOURCES "./*.cpp")

add_library( data ${SOURCES})

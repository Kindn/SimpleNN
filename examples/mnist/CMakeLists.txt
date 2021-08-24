include_directories( ${CMAKE_SOURCE_DIR}/models
                     ${CMAKE_SOURCE_DIR}/util
                     ${CMAKE_SOURCE_DIR}/include ) 

#link_directories(${CMAKE_SOURCE_DIR}/build/src/functions
#${CMAKE_SOURCE_DIR}/build/src/layers
#${CMAKE_SOURCE_DIR}/build/src/optimizers
#${CMAKE_SOURCE_DIR}/build/src/nets
#${CMAKE_SOURCE_DIR}/build/src/loss)

file ( GLOB SOURCES "./*.cpp")

add_executable(mnist_train train.cpp)
target_link_libraries(mnist_train nets layers loss optimizers functions data) # Watch out the link sequence

find_package(OpenCV)
if (OpenCV_FOUND) 
    message("Found OpenCV.")
    include_directories(${OpenCV_INCLUDE_DIRS})                    
    add_executable(mnist_test test.cpp)
    target_link_libraries(mnist_test nets layers loss optimizers functions data ${OpenCV_LIBS})
    add_executable(recognition recognition.cpp)
    target_link_libraries(recognition nets layers loss optimizers functions data ${OpenCV_LIBS})
    add_executable(recognition_camera recognition_camera.cpp)
    target_link_libraries(recognition_camera nets layers loss optimizers functions data ${OpenCV_LIBS})
else(OpenCV_FOUND)
    message("OpenCV not found.Some examples will not be built.")
endif()

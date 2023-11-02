cmake_minimum_required(VERSION 3.18)

project(detection
        VERSION 0.0.1
        DESCRIPTION "Dnn Yolo Detection sample"
        LANGUAGES C CXX CUDA)
set(PROJECT_NAME_INTERNAL ${PROJECT_NAME})
        

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-function")

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/libs")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/libs" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/libs")
ENDIF("${isSystemDir}" STREQUAL "-1")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} "") 

target_sources(${PROJECT_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/SafeQueue.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Detector.h
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Detector.cpp            
        )

target_link_libraries(${PROJECT_NAME}
        -lrt
        -lpthread
        ${OpenCV_LIBS}
        nvinfer_plugin
        nvinfer
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
        CUDA::toolkit
)

target_include_directories( ${PROJECT_NAME} PRIVATE 
                                            ${CMAKE_CURRENT_SOURCE_DIR}/include/
                                            )
#add_version(${PROJECT_NAME} GIT_VERSION)
#install_lib_with_deps( ${PROJECT_NAME} )

SET(SUBPROJECT_NAME SupervisedDescentModel)
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc features2d nonfree)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
FIND_PACKAGE(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# source and header files
SET(HEADERS
	include/superviseddescentmodel/SdmLandmarkModel.hpp
	include/superviseddescentmodel/LandmarkBasedSupervisedDescentTraining.hpp
	include/superviseddescentmodel/DescriptorExtractor.hpp
	include/superviseddescentmodel/hog.h
)
SET(SOURCE
	src/superviseddescentmodel/SdmLandmarkModel.cpp
	src/superviseddescentmodel/LandmarkBasedSupervisedDescentTraining.cpp
	src/superviseddescentmodel/DescriptorExtractor.cpp
	src/superviseddescentmodel/hog.c
)

include_directories("include")

# add dependencies
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
#include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${Logging_SOURCE_DIR}/include)

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ${OpenCV_LIBS} ${Boost_LIBRARIES})

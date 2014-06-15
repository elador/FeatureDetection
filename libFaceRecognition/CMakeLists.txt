set(SUBPROJECT_NAME FaceRecognition)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

#find_package(OpenCV 2.4.3 REQUIRED core)

# source and header files
set(HEADERS
	include/facerecognition/FaceRecord.hpp
	include/facerecognition/utils.hpp
)
set(SOURCE
	src/facerecognition/FaceRecord.cpp
	src/facerecognition/utils.cpp
)

include_directories("include")

# add dependencies
include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${Logging_SOURCE_DIR}/include)

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ${Boost_LIBRARIES}) # ${OpenCV_LIBS} ImageIO

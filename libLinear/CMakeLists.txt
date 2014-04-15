set(SUBPROJECT_NAME Linear)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(BLAS REQUIRED)

find_package(OpenCV 2.4.3 REQUIRED core)

# source and header files
set(HEADERS
	include/tron.h
	include/linear.h
	include/liblinear/LibLinearClassifier.hpp
	include/liblinear/LibLinearUtils.hpp
)
set(SOURCE
	src/tron.cpp
	src/linear.cpp
	src/liblinear/LibLinearClassifier.cpp
	src/liblinear/LibLinearUtils.cpp
)

include_directories("include")

# add dependencies
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${BLAS_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} ${BLAS_LIBRARIES} ${OpenCV_LIBS})

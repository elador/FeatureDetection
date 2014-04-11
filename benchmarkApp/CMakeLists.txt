SET( SUBPROJECT_NAME benchmarkApp ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(BLAS REQUIRED)

FIND_PACKAGE(Boost 1.48.0 COMPONENTS random system filesystem program_options REQUIRED)

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc highgui video)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
# Matlab
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
FIND_PACKAGE(MATLAB REQUIRED)

if(WITH_MSKINECT_SDK)
	# Include Microsoft Kinect SDK (Windows)
	set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
	find_package(KinectSDK REQUIRED)
	include_directories(${KinectSDK_INCLUDE_DIR})
	set(KINECT_LIBNAME ${KinectSDK_LIBRARIES})
	add_definitions(-DWITH_MSKINECT_SDK)
else()
	set(KINECT_LIBNAME "")
endif()

# source and header files
SET(HEADERS
	AlgorithmData.hpp
	Benchmark.hpp
)
SET(SOURCE
	AlgorithmData.cpp
	Benchmark.cpp
	BenchmarkRunner.cpp
)

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${SVM_SOURCE_DIR}/include )
include_directories( ${Linear_SOURCE_DIR}/include )
include_directories( ${Detection_SOURCE_DIR}/include )

# make executable
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Detection Linear SVM Classification ImageProcessing ImageIO ${KINECT_LIBNAME} Logging ${OpenCV_LIBS} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${Boost_LIBRARIES} ${BLAS_LIBRARIES} )

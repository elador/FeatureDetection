SET( SUBPROJECT_NAME Detection ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
# Boost:
FIND_PACKAGE(Boost 1.48.0 COMPONENTS program_options REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()
# OpenCV:
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/detection/ClassifiedPatch.hpp
	include/detection/Detector.hpp
	include/detection/SlidingWindowDetector.hpp
	include/detection/OverlapElimination.hpp
	include/detection/FiveStageSlidingWindowDetector.hpp
)
SET(SOURCE
	src/detection/SlidingWindowDetector.cpp
	src/detection/OverlapElimination.cpp
	src/detection/FiveStageSlidingWindowDetector.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageLogging_SOURCE_DIR}/include )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

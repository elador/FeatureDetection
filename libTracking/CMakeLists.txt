SET( SUBPROJECT_NAME Tracking ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/tracking/CondensationTracker.h
	include/tracking/DensityPositionExtractor.h
	include/tracking/DualSvmMeasurementModel.h
	include/tracking/LowVarianceSampling.h
	include/tracking/MeasurementModel.h
	include/tracking/PositionExtractor.h
	include/tracking/Rectangle.h
	include/tracking/ResamplingAlgorithm.h
	include/tracking/Sample.h
	include/tracking/SimpleTransitionModel.h
	include/tracking/TransitionModel.h
	include/tracking/WeightedMeanPositionExtractor.h
)
SET(SOURCE
	src/tracking/CondensationTracker.cpp
	src/tracking/DensityPositionExtractor.cpp
	src/tracking/DualSvmMeasurementModel.cpp
	src/tracking/LowVarianceSampling.cpp
	src/tracking/Rectangle.cpp
	src/tracking/Sample.cpp
	src/tracking/SimpleTransitionModel.cpp
	src/tracking/WeightedMeanPositionExtractor.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make library
add_library( ${SUBPROJECT_NAME} SHARED ${SOURCE} ${HEADERS} )


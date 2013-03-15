SET( SUBPROJECT_NAME Condensation ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/condensation/AdaptiveCondensationTracker.h
	include/condensation/AdaptiveMeasurementModel.h
	include/condensation/CondensationTracker.h
	include/condensation/DensityPositionExtractor.h
	include/condensation/FilteringPositionExtractor.h
	include/condensation/GridSampler.h
	include/condensation/LowVarianceSampling.h
	include/condensation/MaxWeightPositionExtractor.h
	include/condensation/MeasurementModel.h
	include/condensation/PositionDependentMeasurementModel.h
	include/condensation/PositionExtractor.h
	include/condensation/Rectangle.h
	include/condensation/ResamplingAlgorithm.h
	include/condensation/ResamplingSampler.h
	include/condensation/Sample.h
	include/condensation/Sampler.h
	include/condensation/SelfLearningMeasurementModel.h
	include/condensation/SimpleTransitionModel.h
	include/condensation/TransitionModel.h
	include/condensation/WeightedMeanPositionExtractor.h
	include/condensation/WvmSvmModel.h
)
SET(SOURCE
	src/condensation/AdaptiveCondensationTracker.cpp
	src/condensation/CondensationTracker.cpp
	src/condensation/DensityPositionExtractor.cpp
	src/condensation/FilteringPositionExtractor.cpp
	src/condensation/GridSampler.cpp
	src/condensation/LowVarianceSampling.cpp
	src/condensation/MaxWeightPositionExtractor.cpp
	src/condensation/PositionDependentMeasurementModel.cpp
	src/condensation/ResamplingSampler.cpp
	src/condensation/SelfLearningMeasurementModel.cpp
	src/condensation/SimpleTransitionModel.cpp
	src/condensation/WeightedMeanPositionExtractor.cpp
	src/condensation/WvmSvmModel.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${Detection_SOURCE_DIR}/include )
include_directories( ${Boost_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

SET( SUBPROJECT_NAME Condensation ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost 1.48.0 REQUIRED)
FIND_PACKAGE(OpenCV 2.3.1 REQUIRED core)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/condensation/AdaptiveMeasurementModel.hpp
	include/condensation/CondensationTracker.hpp
	include/condensation/FilteringPositionExtractor.hpp
	include/condensation/GridSampler.hpp
	include/condensation/LowVarianceSampling.hpp
	include/condensation/MaxWeightPositionExtractor.hpp
	include/condensation/MeasurementModel.hpp
	include/condensation/PartiallyAdaptiveCondensationTracker.hpp
	include/condensation/PositionDependentMeasurementModel.hpp
	include/condensation/PositionExtractor.hpp
	include/condensation/ResamplingAlgorithm.hpp
	include/condensation/ResamplingSampler.hpp
	include/condensation/Sample.hpp
	include/condensation/Sampler.hpp
	include/condensation/SelfLearningMeasurementModel.hpp
	include/condensation/SimpleTransitionModel.hpp
	include/condensation/SingleClassifierModel.hpp
	include/condensation/TransitionModel.hpp
	include/condensation/WeightedMeanPositionExtractor.hpp
	include/condensation/WvmSvmModel.hpp
)
SET(SOURCE
	src/condensation/CondensationTracker.cpp
	src/condensation/FilteringPositionExtractor.cpp
	src/condensation/GridSampler.cpp
	src/condensation/LowVarianceSampling.cpp
	src/condensation/MaxWeightPositionExtractor.cpp
	src/condensation/PartiallyAdaptiveCondensationTracker.cpp
	src/condensation/PositionDependentMeasurementModel.cpp
	src/condensation/ResamplingSampler.cpp
	src/condensation/SelfLearningMeasurementModel.cpp
	src/condensation/SimpleTransitionModel.cpp
	src/condensation/SingleClassifierModel.cpp
	src/condensation/WeightedMeanPositionExtractor.cpp
	src/condensation/WvmSvmModel.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${Detection_SOURCE_DIR}/include )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

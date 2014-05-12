set(SUBPROJECT_NAME Condensation)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set( ${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set( ${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 REQUIRED)

find_package(OpenCV 2.4.3 REQUIRED core imgproc video)

# source and header files
set(HEADERS
	include/condensation/AdaptiveCondensationTracker.hpp
	include/condensation/AdaptiveMeasurementModel.hpp
	include/condensation/ClassificationBasedStateValidator.hpp
	include/condensation/CondensationTracker.hpp
	include/condensation/ExtendedHogBasedMeasurementModel.hpp
	include/condensation/FilteringClassifierModel.hpp
	include/condensation/FilteringStateExtractor.hpp
	include/condensation/GridSampler.hpp
	include/condensation/LowVarianceSampling.hpp
	include/condensation/MaxWeightStateExtractor.hpp
	include/condensation/MeasurementModel.hpp
	include/condensation/OpticalFlowTransitionModel.hpp
	include/condensation/PartiallyAdaptiveCondensationTracker.hpp
	include/condensation/PositionDependentMeasurementModel.hpp
	include/condensation/ResamplingAlgorithm.hpp
	include/condensation/ResamplingSampler.hpp
	include/condensation/Sample.hpp
	include/condensation/Sampler.hpp
	include/condensation/SelfLearningMeasurementModel.hpp
	include/condensation/SimpleTransitionModel.hpp
	include/condensation/SingleClassifierModel.hpp
	include/condensation/StateExtractor.hpp
	include/condensation/StateValidator.hpp
	include/condensation/TransitionModel.hpp
	include/condensation/WeightedMeanStateExtractor.hpp
	include/condensation/WvmSvmModel.hpp
)
set(SOURCE
	src/condensation/AdaptiveCondensationTracker.cpp
	src/condensation/ClassificationBasedStateValidator.cpp
	src/condensation/CondensationTracker.cpp
	src/condensation/ExtendedHogBasedMeasurementModel.cpp
	src/condensation/FilteringClassifierModel.cpp
	src/condensation/FilteringStateExtractor.cpp
	src/condensation/GridSampler.cpp
	src/condensation/LowVarianceSampling.cpp
	src/condensation/MaxWeightStateExtractor.cpp
	src/condensation/OpticalFlowTransitionModel.cpp
	src/condensation/PartiallyAdaptiveCondensationTracker.cpp
	src/condensation/PositionDependentMeasurementModel.cpp
	src/condensation/ResamplingSampler.cpp
	src/condensation/Sample.cpp
	src/condensation/SelfLearningMeasurementModel.cpp
	src/condensation/SimpleTransitionModel.cpp
	src/condensation/SingleClassifierModel.cpp
	src/condensation/WeightedMeanStateExtractor.cpp
	src/condensation/WvmSvmModel.cpp
)

include_directories("include")

# add dependencies
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${Detection_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Classification ImageProcessing ${Boost_LIBRARIES} ${OpenCV_LIBS})

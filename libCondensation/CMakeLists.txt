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
	include/condensation/CondensationTracker.hpp
	include/condensation/ConstantVelocityModel.hpp
	include/condensation/ExtendedHogBasedMeasurementModel.hpp
	include/condensation/FilteringStateExtractor.hpp
	include/condensation/LowVarianceSampling.hpp
	include/condensation/MaxWeightStateExtractor.hpp
	include/condensation/MeasurementModel.hpp
	include/condensation/OpticalFlowTransitionModel.hpp
	include/condensation/ResamplingAlgorithm.hpp
	include/condensation/Sample.hpp
	include/condensation/StateExtractor.hpp
	include/condensation/StateValidator.hpp
	include/condensation/TransitionModel.hpp
	include/condensation/WeightedMeanStateExtractor.hpp
)
set(SOURCE
	src/condensation/CondensationTracker.cpp
	src/condensation/ConstantVelocityModel.cpp
	src/condensation/ExtendedHogBasedMeasurementModel.cpp
	src/condensation/FilteringStateExtractor.cpp
	src/condensation/LowVarianceSampling.cpp
	src/condensation/MaxWeightStateExtractor.cpp
	src/condensation/OpticalFlowTransitionModel.cpp
	src/condensation/Sample.cpp
	src/condensation/WeightedMeanStateExtractor.cpp
)

include_directories("include")

# add dependencies
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Classification ImageProcessing ${Boost_LIBRARIES} ${OpenCV_LIBS})

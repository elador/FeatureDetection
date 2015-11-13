SET( SUBPROJECT_NAME Detection ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(Boost 1.48.0 COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/detection/AggregatedFeaturesDetector.hpp
	include/detection/ClassifiedPatch.hpp
	include/detection/Detector.hpp
	include/detection/FiveStageSlidingWindowDetector.hpp
	include/detection/NonMaximumSuppression.hpp
	include/detection/OverlapElimination.hpp
	include/detection/SimpleDetector.hpp
	include/detection/SlidingWindowDetector.hpp
)
SET(SOURCE
	src/detection/AggregatedFeaturesDetector.cpp
	src/detection/FiveStageSlidingWindowDetector.cpp
	src/detection/NonMaximumSuppression.cpp
	src/detection/OverlapElimination.cpp
	src/detection/SlidingWindowDetector.cpp
)

include_directories("include")

# add dependencies
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageLogging_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ImageLogging ImageProcessing ${Boost_LIBRARIES} ${OpenCV_LIBS})

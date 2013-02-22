SET( SUBPROJECT_NAME ShapeModels ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/shapemodels/FeaturePointsModel.hpp
	include/shapemodels/FeaturePointsSelector.hpp
	include/shapemodels/ListFeaturePointsSelector.hpp
	include/shapemodels/RansacFeaturePointsSelector.hpp
	include/shapemodels/FeaturePointsEvaluator.hpp
	include/shapemodels/TwoStepFeaturePointsEvaluator.hpp
	include/shapemodels/FastProjectionFeaturePointsEvaluator.hpp
	include/shapemodels/FeaturePointsEvaluator3dmm.hpp
)
SET(SOURCE
	src/shapemodels/RansacFeaturePointsSelector.cpp
)

include_directories( "include" )

# add dependencies
#include_directories( ${Boost_INCLUDE_DIRS} )
#include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

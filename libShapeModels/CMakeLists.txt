SET( SUBPROJECT_NAME ShapeModels ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost 1.48.0 REQUIRED)
FIND_PACKAGE(OpenCV 2.3.1 REQUIRED core calib3d)

# source and header files
SET(HEADERS
	include/shapemodels/PcaModel.hpp
	include/shapemodels/MorphableModel.hpp
	include/shapemodels/CameraEstimation.hpp
	include/shapemodels/FeaturePointsModelRANSACtmp.hpp
	include/shapemodels/FeaturePointsModel.hpp
	include/shapemodels/FeaturePointsSelector.hpp
	include/shapemodels/ListFeaturePointsSelector.hpp
	include/shapemodels/RandomFeaturePointsSelector.hpp
	include/shapemodels/FeaturePointsEvaluator.hpp
	include/shapemodels/TwoStepFeaturePointsEvaluator.hpp
	include/shapemodels/FastProjectionFeaturePointsEvaluator.hpp
	include/shapemodels/FeaturePointsEvaluator3dmm.hpp
	include/shapemodels/PositFeaturePointsEvaluator.hpp
	include/shapemodels/RansacFeaturePointsModel.hpp
)
SET(SOURCE
	src/shapemodels/PcaModel.cpp
	src/shapemodels/MorphableModel.cpp
	src/shapemodels/CameraEstimation.cpp
	src/shapemodels/RandomFeaturePointsSelector.cpp
	src/shapemodels/FeaturePointsModelRANSACtmp.cpp
	src/shapemodels/PositFeaturePointsEvaluator.cpp
	src/shapemodels/RansacFeaturePointsModel.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES("C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\include\\hdf5\\")
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${Render_SOURCE_DIR}/include )
include_directories( ${Logging_SOURCE_DIR}/include )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

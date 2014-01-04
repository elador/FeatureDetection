SET( SUBPROJECT_NAME ShapeModels ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost 1.48.0 REQUIRED)
FIND_PACKAGE(OpenCV 2.3.1 REQUIRED core calib3d)

#IF(WIN32)
	#FIND_PACKAGE (HDF5 NAMES hdf5) # puts it in Config-mode
	#FIND_PACKAGE (HDF5 REQUIRED CONFIG) # same
	FIND_PACKAGE (HDF5 REQUIRED COMPONENTS C CXX) # works on windows as well
#ENDIF()
#message(STATUS "Hdf5 include dir: ${HDF5_INCLUDE_DIR}") # According to hdf5 cmake file, should use *_DIRS, but they don't define it in hdf5-config.cmake. Ok if find_packaged used as above (now), it works.
message(STATUS "HDF5 includes at: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries at: ${HDF5_LIBRARIES}")

# source and header files
SET(HEADERS
	include/shapemodels/PcaModel.hpp
	include/shapemodels/MorphableModel.hpp
	include/shapemodels/CameraEstimation.hpp # Should rename to OpenCvCameraEstimation
	include/shapemodels/AffineCameraEstimation.hpp
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
	src/shapemodels/AffineCameraEstimation.cpp
	src/shapemodels/RandomFeaturePointsSelector.cpp
	src/shapemodels/FeaturePointsModelRANSACtmp.cpp
	src/shapemodels/PositFeaturePointsEvaluator.cpp
	src/shapemodels/RansacFeaturePointsModel.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${Render_SOURCE_DIR}/include )
include_directories( ${Logging_SOURCE_DIR}/include )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

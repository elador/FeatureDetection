SET( SUBPROJECT_NAME MorphableModel ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(Boost 1.48.0 REQUIRED)
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)

#Hdf5:
IF(WIN32)
	#FIND_PACKAGE (HDF5 NAMES hdf5) # puts it in Config-mode
	FIND_PACKAGE (HDF5 REQUIRED COMPONENTS C CXX NAMES hdf5) # COMPONENTS doesn't seem to work
	set (HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR}) # According to hdf5 cmake file, should use *_DIRS, but they don't define it in hdf5-config.cmake. So if we use the Config-mode with NAMES, it isn't set, but it is, when used as in linux.
ELSE()
	#FIND_PACKAGE (HDF5 REQUIRED CONFIG) # same
	FIND_PACKAGE (HDF5 REQUIRED COMPONENTS C CXX) # works on windows as well but links to dll's instead of lib's...
ENDIF()
message(STATUS "HDF5 includes at: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries at: ${HDF5_LIBRARIES}")

SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
FIND_PACKAGE(Eigen3 REQUIRED)
MESSAGE(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
MESSAGE(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
MESSAGE(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# source and header files
SET(HEADERS
	include/morphablemodel/PcaModel.hpp
	include/morphablemodel/MorphableModel.hpp
	include/morphablemodel/OpenCVCameraEstimation.hpp
	include/morphablemodel/AffineCameraEstimation.hpp
	include/morphablemodel/Hdf5Utils.hpp
)
SET(SOURCE
	src/morphablemodel/PcaModel.cpp
	src/morphablemodel/MorphableModel.cpp
	src/morphablemodel/OpenCVCameraEstimation.cpp
	src/morphablemodel/AffineCameraEstimation.cpp
	src/morphablemodel/Hdf5Utils.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ${HDF5_INCLUDE_DIRS} )
include_directories( ${Render_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${Logging_SOURCE_DIR}/include )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

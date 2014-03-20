SET( SUBPROJECT_NAME Render )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

#OpenCV:
FIND_PACKAGE( OpenCV 2.4.3 REQUIRED core imgproc highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Boost:
FIND_PACKAGE(Boost 1.48.0 COMPONENTS random REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

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

#Source and header files:
SET(SOURCE
	src/render/SoftwareRenderer.cpp
	src/render/Vertex.cpp
	src/render/Triangle.cpp
	src/render/Camera.cpp
	src/render/Texture.cpp
	src/render/Mesh.cpp
	src/render/MatrixUtils.cpp
	src/render/MeshUtils.cpp
	src/render/Hdf5Utils.cpp
)

SET(HEADERS
	include/render/SoftwareRenderer.hpp
	include/render/Vertex.hpp
	include/render/Triangle.hpp
	include/render/Camera.hpp
	include/render/Texture.hpp
	include/render/Mesh.hpp
	include/render/MatrixUtils.hpp
	include/render/MeshUtils.hpp
	include/render/Hdf5Utils.hpp
)

include_directories( "include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

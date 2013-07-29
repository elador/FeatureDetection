SET( SUBPROJECT_NAME Renderold )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

#OpenCV:
FIND_PACKAGE( OpenCV 2.3.1 REQUIRED core imgproc highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Boost:
FIND_PACKAGE(Boost 1.48.0 COMPONENTS random REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

#HDF5: (needed for reading the Model)
#MESSAGE(STATUS "HDF5_ROOT: ${HDF5_ROOT}")
#SET( CMAKE_MODULE_PATH ${HDF5_ROOT}/cmake/hdf5 )
#MESSAGE(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
#FIND_PACKAGE( HDF5 REQUIRED COMPONENTS C CXX )
#FIND_PACKAGE( HDF5 NAMES hdf5 )
#SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
#FIND_PACKAGE( HDF5 NAMES hdf5 )

#Source and header files:
SET(SOURCE
	src/renderold/SRenderer.cpp
	src/renderold/Vertex.cpp
	src/renderold/Triangle.cpp
	src/renderold/Camera.cpp
	src/renderold/Texture.cpp
	src/renderold/Mesh.cpp
	src/renderold/MatrixUtils.cpp
	src/renderold/MeshUtils.cpp
	src/renderold/Hdf5Utils.cpp
	src/renderold/MorphableModel.cpp
)

SET(HEADERS
	include/renderold/SRenderer.hpp
	include/renderold/Vertex.hpp
	include/renderold/Triangle.hpp
	include/renderold/Camera.hpp
	include/renderold/Texture.hpp
	include/renderold/Mesh.hpp
	include/renderold/MatrixUtils.hpp
	include/renderold/MeshUtils.hpp
	include/renderold/Hdf5Utils.hpp
	include/renderold/MorphableModel.hpp
)

include_directories( "include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\include\\hdf5\\")

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

SET( SUBPROJECT_NAME Render )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

#OpenCV:
FIND_PACKAGE( OpenCV REQUIRED core imgproc highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Boost:
FIND_PACKAGE(Boost COMPONENTS random REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

#HDF5: (needed for reading the Model)
FIND_PACKAGE( HDF5 REQUIRED COMPONENTS C CXX )
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
MESSAGE(STATUS "HDF5 include dir: ${HDF5_INCLUDE_DIRS}")

#Source and header files:
SET(SOURCE
	src/render/SRenderer.cpp
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
	include/render/SRenderer.hpp
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

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

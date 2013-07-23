SET( SUBPROJECT_NAME Render2 )	# my own variable, not cmake's
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
	src/render2/Renderer.cpp
	src/render2/OpenGlDevice.cpp
	src/render2/SoftwareDevice.cpp
	src/render2/Vertex.cpp
	src/render2/Triangle.cpp
	src/render2/Camera.cpp
	src/render2/Texture.cpp
	src/render2/Mesh.cpp
	src/render2/MatrixUtils.cpp
	src/render2/MeshUtils.cpp
	src/render2/Hdf5Utils.cpp
	src/render2/MorphableModel.cpp
)

SET(HEADERS
	include/render2/Renderer.hpp
	include/render2/RenderDevice.hpp
	include/render2/OpenGlDevice.hpp
	include/render2/SoftwareDevice.hpp
	include/render2/Vertex.hpp
	include/render2/Triangle.hpp
	include/render2/Camera.hpp
	include/render2/Texture.hpp
	include/render2/Mesh.hpp
	include/render2/MatrixUtils.hpp
	include/render2/MeshUtils.hpp
	include/render2/Hdf5Utils.hpp
	include/render2/MorphableModel.hpp
)

include_directories( "include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\include\\hdf5\\")

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

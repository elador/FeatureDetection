SET( SUBPROJECT_NAME Render )	# my own variable, not cmake's
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
	src/render/Renderer.cpp
	src/render/RenderDevice.cpp
	src/render/RenderDevicePnP.cpp
	src/render/OpenGlDevice.cpp
	src/render/SoftwareDevice.cpp
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
	include/render/Renderer.hpp
	include/render/RenderDevice.hpp
	include/render/RenderDevicePnP.hpp
	include/render/OpenGlDevice.hpp
	include/render/SoftwareDevice.hpp
	include/render/Vertex.hpp
	include/render/Triangle.hpp
	include/render/Camera.hpp
	include/render/Texture.hpp
	include/render/Mesh.hpp
	include/render/MatrixUtils.hpp
	include/render/MeshUtils.hpp
	include/render/Hdf5Utils.hpp
	include/render/OGLDevNew.hpp
)

include_directories( "include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\include\\hdf5\\")

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

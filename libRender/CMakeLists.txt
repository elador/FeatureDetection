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
#FIND_PACKAGE(Boost 1.48.0 COMPONENTS random REQUIRED)
FIND_PACKAGE(Boost 1.48.0 REQUIRED)
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

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
)

include_directories( "include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

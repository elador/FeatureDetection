SET( SUBPROJECT_NAME Render )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
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

#set (CMAKE_LIBRARY_PATH "C:\\Program Files (x86)\\Windows Kits\\8.0\\Lib\\win8\\um\\x64")
#set (CMAKE_PREFIX_PATH "C:\\Program Files (x86)\\Windows Kits\\8.0\\Lib\\win8\\um\\x64")
#set (CMAKE_MODULE_PATH "C:\\Program Files (x86)\\Windows Kits\\8.0\\Lib\\win8\\um\\x64")

#find_package(glu32)
#find_package(opengl)
#message("TEST_Qt5Gui_GL_LIBRARY : ${TEST_Qt5Gui_GL_LIBRARY}")
message("CMAKE_CXX_LIBRARY_ARCHITECTURE ${CMAKE_CXX_LIBRARY_ARCHITECTURE}")

set(CMAKE_AUTOMOC ON) # Todo: Probably make dependent on WITH_RENDER_OPENGL
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
#find_package(Qt5OpenGL REQUIRED)
#find_package(Qt5Widgets REQUIRED)

message(status "EGL inc: ${Qt5Gui_EGL_INCLUDE_DIRS}")
message(status "OpenGL inc: ${Qt5Gui_OPENGL_INCLUDE_DIRS}")
message(status "EGL libs: ${Qt5Gui_EGL_LIBRARIES}")
message(status "OpenGL libs: ${Qt5Gui_OPENGL_LIBRARIES}")

#Source and header files:
SET(SOURCE
	src/render/QOpenGLRenderer.cpp
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
	include/render/QOpenGLRenderer.hpp
	include/render/SoftwareRenderer.hpp
	include/render/SoftwareRenderer2.hpp
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

# Required to find the headers. See our github issue.
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Qt5::Core Qt5::Gui ${Qt5Gui_EGL_LIBRARIES} ${Qt5Gui_OPENGL_LIBRARIES} )

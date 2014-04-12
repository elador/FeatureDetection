SET( SUBPROJECT_NAME Render )
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc highgui)

#FIND_PACKAGE(Boost 1.48.0 COMPONENTS random REQUIRED)
FIND_PACKAGE(Boost 1.48.0 REQUIRED)
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

set(CMAKE_AUTOMOC ON) # Todo: Probably make dependent on WITH_RENDER_OPENGL
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
message(STATUS "Qt5 EGL include dir: ${Qt5Gui_EGL_INCLUDE_DIRS}")
message(STATUS "Qt5 OpenGL include dir: ${Qt5Gui_OPENGL_INCLUDE_DIRS}")
message(STATUS "Qt5 EGL library: ${Qt5Gui_EGL_LIBRARIES}")
message(STATUS "Qt5 OpenGL library: ${Qt5Gui_OPENGL_LIBRARIES}")

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
	include/render/Vertex.hpp
	include/render/Triangle.hpp
	include/render/Camera.hpp
	include/render/Texture.hpp
	include/render/Mesh.hpp
	include/render/MatrixUtils.hpp
	include/render/MeshUtils.hpp
)

include_directories("include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

#Make the library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Qt5::Core Qt5::Gui ${Qt5Gui_EGL_LIBRARIES} ${Qt5Gui_OPENGL_LIBRARIES} ${OpenCV_LIBS}) # ${Boost_LIBRARIES}

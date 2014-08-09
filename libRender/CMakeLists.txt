set(SUBPROJECT_NAME Render)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8.11)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)

find_package(Boost 1.48.0 REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

if(WITH_RENDER_QOPENGL)
	set(CMAKE_AUTOMOC ON)
	find_package(Qt5Core REQUIRED)
	find_package(Qt5Gui REQUIRED)
	message(STATUS "Qt5 EGL include dir: ${Qt5Gui_EGL_INCLUDE_DIRS}")
	message(STATUS "Qt5 OpenGL include dir: ${Qt5Gui_OPENGL_INCLUDE_DIRS}")
	message(STATUS "Qt5 EGL library: ${Qt5Gui_EGL_LIBRARIES}")
	message(STATUS "Qt5 OpenGL library: ${Qt5Gui_OPENGL_LIBRARIES}")
	add_definitions(-DWITH_RENDER_QOPENGL)
endif()

# Source and header files:
set(SOURCE
	src/render/QOpenGLRenderer.cpp
	src/render/SoftwareRenderer.cpp
	src/render/Vertex.cpp
	src/render/Triangle.cpp
	src/render/Camera.cpp
	src/render/Texture.cpp
	src/render/Mesh.cpp
	src/render/MatrixUtils.cpp
	src/render/MeshUtils.cpp
	src/render/utils.cpp
)

set(HEADERS
	include/render/QOpenGLRenderer.hpp
	include/render/SoftwareRenderer.hpp
	include/render/Vertex.hpp
	include/render/Triangle.hpp
	include/render/Camera.hpp
	include/render/Texture.hpp
	include/render/Mesh.hpp
	include/render/MatrixUtils.hpp
	include/render/MeshUtils.hpp
	include/render/utils.hpp
)

include_directories("include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Make the library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
if(WITH_RENDER_QOPENGL)
	target_link_libraries(${SUBPROJECT_NAME} Qt5::Core Qt5::Gui ${Qt5Gui_EGL_LIBRARIES} ${Qt5Gui_OPENGL_LIBRARIES} ${OpenCV_LIBS})
else()
	target_link_libraries(${SUBPROJECT_NAME} ${OpenCV_LIBS})
endif()

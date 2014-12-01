#http://www.cmake.org/cmake/help/cmake2.6docs.html
project (OpenGL_Qt_Binding)
cmake_minimum_required (VERSION 2.8.11)

message(STATUS "=== Configuring OpenGL_Qt_Binding ===")

FIND_PACKAGE( OpenCV REQUIRED )
#find_package (OpenGL REQUIRED)

set(CMAKE_AUTOMOC ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
message(STATUS "Qt5 EGL include dir: ${Qt5Gui_EGL_INCLUDE_DIRS}")
message(STATUS "Qt5 OpenGL include dir: ${Qt5Gui_OPENGL_INCLUDE_DIRS}")
message(STATUS "Qt5 EGL library: ${Qt5Gui_EGL_LIBRARIES}")
message(STATUS "Qt5 OpenGL library: ${Qt5Gui_OPENGL_LIBRARIES}")

SET(SOURCE
	main.cpp
	openglwindow.cpp
)

SET(HEADERS
	openglwindow.h
)

ADD_EXECUTABLE(OpenGL_Qt_Binding ${SOURCE} ${HEADERS})

TARGET_LINK_LIBRARIES(OpenGL_Qt_Binding ${OpenCV_LIBS} Qt5::Core Qt5::Gui ${Qt5Gui_EGL_LIBRARIES} ${Qt5Gui_OPENGL_LIBRARIES}) # ${OPENGL_LIBRARIES}
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube4.avi ${CMAKE_CURRENT_BINARY_DIR}/cube4.avi COPYONLY)

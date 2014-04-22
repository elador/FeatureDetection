set(SUBPROJECT_NAME fitter)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8.11)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

#cmake_policy(SET CMP0020 NEW) # all Qt projects. auto-link to winmain... etc. Better: cmake_policy(VERSION 2.8.11) (first version to support auto-linking (and the new Qt imports anyway).
# Todo: move the policy to the sub-project. Because: it gives an error on a cmake version that doesnt support this policy!
# Note: with CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11), the policy is already set to default for 2.8.11 which is ON.

# find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui calib3d objdetect)

find_package(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

set(CMAKE_AUTOMOC ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
message(STATUS "Qt5 EGL include dir: ${Qt5Gui_EGL_INCLUDE_DIRS}")
message(STATUS "Qt5 OpenGL include dir: ${Qt5Gui_OPENGL_INCLUDE_DIRS}")
message(STATUS "Qt5 EGL library: ${Qt5Gui_EGL_LIBRARIES}")
message(STATUS "Qt5 OpenGL library: ${Qt5Gui_OPENGL_LIBRARIES}")

#Source and header files:
set(SOURCE
	fitter.cpp
	OpenGLWindow.cpp
	FittingWindow.cpp
)

set(HEADERS
	OpenGLWindow.hpp
	FittingWindow.hpp
)

add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${Render_SOURCE_DIR}/include)
include_directories(${MorphableModel_SOURCE_DIR}/include)

#Make the app depend on the libraries
target_link_libraries(${SUBPROJECT_NAME} ImageIO ${KINECT_LIBNAME} MorphableModel Render Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} ${HDF5_LIBRARIES} Qt5::Core Qt5::Gui ${Qt5Gui_EGL_LIBRARIES} ${Qt5Gui_OPENGL_LIBRARIES}) # Qt5::OpenGL Qt5::Widgets ${OPENGL_LIBRARIES} ${GLES_LIBRARIES}

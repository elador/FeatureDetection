SET( SUBPROJECT_NAME fitter )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

#cmake_policy(SET CMP0020 NEW) # all Qt projects. auto-link to winmain... etc. Better: cmake_policy(VERSION 2.8.11) (first version to support auto-linking (and the new Qt imports anyway).
# Todo: move the policy to the sub-project. Because: it gives an error on a cmake version that doesnt support this policy!
# Note: with CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11), the policy is already set to default for 2.8.11 which is ON.

#OpenCV:
FIND_PACKAGE( OpenCV 2.4.3 REQUIRED core imgproc highgui calib3d objdetect )
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Hdf5:
IF(WIN32)
	#FIND_PACKAGE (HDF5 NAMES hdf5) # puts it in Config-mode
	FIND_PACKAGE (HDF5 REQUIRED COMPONENTS C CXX NAMES hdf5) # COMPONENTS doesn't seem to work
	set (HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR}) # According to hdf5 cmake file, should use *_DIRS, but they don't define it in hdf5-config.cmake. So if we use the Config-mode with NAMES, it isn't set, but it is, when used as in linux.
ELSE()
	#FIND_PACKAGE (HDF5 REQUIRED CONFIG) # same
	FIND_PACKAGE (HDF5 REQUIRED COMPONENTS C CXX) # works on windows as well but links to dll's instead of lib's...
ENDIF()
message(STATUS "HDF5 includes at: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries at: ${HDF5_LIBRARIES}")

#Matlab:
SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
FIND_PACKAGE( MATLAB REQUIRED )
IF(MATLAB_FOUND)
  MESSAGE(STATUS "Matlab found at ${MATLAB_INCLUDE_DIR}")
ELSE(MATLAB_FOUND)
  MESSAGE(FATAL_ERROR "Matlab not found")
ENDIF(MATLAB_FOUND)

#Boost:
FIND_PACKAGE(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

set(CMAKE_AUTOMOC ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)

message(status "EGL inc: ${Qt5Gui_EGL_INCLUDE_DIRS}")
message(status "OpenGL inc: ${Qt5Gui_OPENGL_INCLUDE_DIRS}")
message(status "EGL libs: ${Qt5Gui_EGL_LIBRARIES}")
message(status "OpenGL libs: ${Qt5Gui_OPENGL_LIBRARIES}")

#Source and header files:
SET(SOURCE
	fitter.cpp
	OpenGLWindow.cpp
	FittingWindow.cpp
)

SET(HEADERS
	OpenGLWindow.hpp
	FittingWindow.hpp
)

ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${Render_SOURCE_DIR}/include )
include_directories( ${MorphableModel_SOURCE_DIR}/include )

#Make the app depend on the libraries
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} ImageIO ${KINECT_LIBNAME} MorphableModel Render Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} ${HDF5_LIBRARIES} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::Widgets ${Qt5Gui_EGL_LIBRARIES} ${Qt5Gui_OPENGL_LIBRARIES} ) # Qt5::OpenGL Qt5::Widgets ${OPENGL_LIBRARIES} ${GLES_LIBRARIES}

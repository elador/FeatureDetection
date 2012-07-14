SET( SUBPROJECT_NAME libKinectInterface )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

#OpenCV:
#FIND_PACKAGE( OpenCV REQUIRED core imgproc highgui)
#MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
#MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Matlab:
#SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
#FIND_PACKAGE( MATLAB REQUIRED )
#IF(MATLAB_FOUND)
#  MESSAGE(STATUS "Matlab found at ${MATLAB_INCLUDE_DIR}")
#ELSE(MATLAB_FOUND)
#  MESSAGE(FATAL_ERROR "Matlab not found")
#ENDIF(MATLAB_FOUND)
#INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

#Source and header files:
SET(SOURCE
	KinectInterface.cpp
	KinectWindowsMSSDK.cpp
	stdafx.cpp
)

SET(HEADERS
	KinectInterface.h
	KinectWindowsMSSDK.h
	stdafx.h
	targetver.h
)

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

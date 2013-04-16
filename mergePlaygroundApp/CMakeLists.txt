SET( SUBPROJECT_NAME mergePlaygroundApp )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

FIND_PACKAGE( OpenCV 2.3.1 REQUIRED core imgproc highgui )
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
FIND_PACKAGE( MATLAB REQUIRED )
FIND_PACKAGE( Boost 1.48.0 COMPONENTS program_options REQUIRED ) # others (e.g.): system filesystem

#Source and header files:
SET(SOURCE
	mergePlaygroundApp.cpp
)

SET(HEADERS
)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})
include_directories( ${OpenCV_INCLUDE_DIRS} )

#Make the library
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

#MESSAGE(STATUS "Adding includes and lib: ${Classification_SOURCE_DIR}/include")
#MESSAGE(STATUS "Adding includes and lib: ${Classification_INCLUDE_DIR}")
include_directories( "${Classification_SOURCE_DIR}/include" )
include_directories( "${ImageProcessing_SOURCE_DIR}/include" )
include_directories( "${Detection_SOURCE_DIR}/include" )
include_directories( "${Logging_SOURCE_DIR}/include" )
#Make the app depend on the libraries
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Classification ImageProcessing Detection Logging ${OpenCV_LIBS} ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} )

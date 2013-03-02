SET( SUBPROJECT_NAME mergePlaygroundApp )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

FIND_PACKAGE( OpenCV REQUIRED core imgproc highgui )
SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
FIND_PACKAGE( MATLAB REQUIRED )
FIND_PACKAGE( Boost COMPONENTS program_options REQUIRED ) # others (e.g.): system filesystem

#Source and header files:
SET(SOURCE
	mergePlaygroundApp.cpp
)

SET(HEADERS
)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

#Make the library
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

#MESSAGE(STATUS "Adding includes and lib: ${Classification_SOURCE_DIR}/include")
#MESSAGE(STATUS "Adding includes and lib: ${Classification_INCLUDE_DIR}")
include_directories( "${Classification_SOURCE_DIR}/include" )
include_directories( "${ImageProcessing_SOURCE_DIR}/include" )
include_directories( "${Detection_SOURCE_DIR}/include" )
include_directories( "${Logging_SOURCE_DIR}/include" )
include_directories( "${ConfigParser_SOURCE_DIR}/include" )
#Make the app depend on the libraries
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Classification ImageProcessing Detection Logging ConfigParser ${OpenCV_LIBS} ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} )

SET( SUBPROJECT_NAME renderTestApp )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

#OpenCV:
FIND_PACKAGE( OpenCV REQUIRED core imgproc highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Boost:
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

#HDF5:
FIND_PACKAGE( HDF5 REQUIRED COMPONENTS C CXX )

#Source and header files:
SET(SOURCE
	renderTestApp.cpp
)

SET(HEADERS
)

include_directories( ${Render_SOURCE_DIR}/include )

#Make the library
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

#MESSAGE(STATUS "Adding includes and lib: ${libFeatureDetection_SOURCE_DIR}")
#include_directories( ${libFeatureDetection_SOURCE_DIR} )
#Make the app depend on the libraries
#TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} libFeatureDetection ${OpenCV_LIBS} ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${HDF5_LIBRARIES} )
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Render ${OpenCV_LIBS} ${Boost_LIBRARIES} )
#MESSAGE(STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")
#MESSAGE(STATUS "HDF5_C_LIBRARIES: ${HDF5_C_LIBRARIES}")
#MESSAGE(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")

SET( SUBPROJECT_NAME faceRecogTestApp )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}:")
MESSAGE(STATUS "-------------------------------")

#OpenCV:
FIND_PACKAGE( OpenCV 2.3.1 REQUIRED core imgproc highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Matlab:
SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
FIND_PACKAGE( MATLAB REQUIRED )
IF(MATLAB_FOUND)
  MESSAGE(STATUS "Matlab found at ${MATLAB_INCLUDE_DIR}")
ELSE(MATLAB_FOUND)
  MESSAGE(FATAL_ERROR "Matlab not found")
ENDIF(MATLAB_FOUND)
INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

#Boost:
FIND_PACKAGE(Boost 1.48.0 COMPONENTS program_options filesystem system REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

#HDF5:
#MESSAGE(STATUS "HDF5_ROOT: ${HDF5_ROOT}")
#SET( CMAKE_MODULE_PATH ${HDF5_ROOT}/cmake/hdf5 )
#MESSAGE(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
#FIND_PACKAGE( HDF5 REQUIRED COMPONENTS C CXX ) # Maybe: FIND_PACKAGE (HDF5 REQURIED), and then, further up, PROJECT (HDF5MyApp C CXX) ? (according to hdf5 docu)


#Source and header files:
SET(SOURCE
	faceRecogTestApp.cpp
)

SET(HEADERS
)

#Make the library
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

INCLUDE_DIRECTORIES( ${MATLAB_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${Detection_SOURCE_DIR}/include )

#Make the app depend on the libraries
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Detection ImageIO ${KINECT_LIBNAME} ImageProcessing Classification Logging ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${OpenCV_LIBS} ) # ${HDF5_LIBRARIES}
#MESSAGE(STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")
#MESSAGE(STATUS "HDF5_C_LIBRARIES: ${HDF5_C_LIBRARIES}")
#MESSAGE(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")

MESSAGE(STATUS "Finished configuring ${SUBPROJECT_NAME}")
MESSAGE(STATUS "===============================")

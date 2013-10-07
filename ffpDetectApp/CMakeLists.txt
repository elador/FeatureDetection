SET( SUBPROJECT_NAME ffpDetectApp )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

#OpenCV:
FIND_PACKAGE( OpenCV 2.3.1 REQUIRED core imgproc highgui calib3d)
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

#Boost:
FIND_PACKAGE(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

#Source and header files:
SET(SOURCE
	ffpDetectApp.cpp
)

SET(HEADERS
)

ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

INCLUDE_DIRECTORIES( ${MATLAB_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageLogging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${Render_SOURCE_DIR}/include )
include_directories( ${ShapeModels_SOURCE_DIR}/include )
include_directories( ${Detection_SOURCE_DIR}/include )

# Hdf5 "hack":
INCLUDE_DIRECTORIES("C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\include\\hdf5\\")
link_directories( "C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\lib\\hdf5\\" )
set(HDF5_LIBRARIES debug hdf5d debug hdf5_cppd optimized hdf5; optimized hdf5_cpp)

#Make the app depend on the libraries
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Detection ImageIO ${KINECT_LIBNAME} ImageProcessing Classification ShapeModels Render ImageLogging Logging ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${OpenCV_LIBS} ${HDF5_LIBRARIES} )

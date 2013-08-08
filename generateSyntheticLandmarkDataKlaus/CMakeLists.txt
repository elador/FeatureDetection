SET( SUBPROJECT_NAME generateSyntheticLandmarkDataKlaus )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

#OpenCV:
FIND_PACKAGE( OpenCV 2.3.1 REQUIRED core imgproc highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

#Boost:
FIND_PACKAGE(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED) # system filesystem for libImageIO (LM stuff)
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

#HDF5:
#FIND_PACKAGE( HDF5 REQUIRED COMPONENTS C CXX )

#Source and header files:
SET(SOURCE
	generateSyntheticLandmarkDataKlaus.cpp
)

SET(HEADERS
)

include_directories( ${Render_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\include\\hdf5\\")

link_directories( "C:\\Users\\Patrik\\Cloud\\PhD\\HDF5\\lib\\hdf5\\" )
set(HDF5_LIBRARIES debug hdf5d debug hdf5_cppd optimized hdf5; optimized hdf5_cpp)

#Make the library
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

#MESSAGE(STATUS "Adding includes and lib: ${libFeatureDetection_SOURCE_DIR}")
#include_directories( ${libFeatureDetection_SOURCE_DIR} )
#Make the app depend on the libraries
#TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} libFeatureDetection ${OpenCV_LIBS} ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${HDF5_LIBRARIES} )
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Render ImageIO ${OpenCV_LIBS} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} )
MESSAGE(STATUS "APP HDF5 lib dir: ${HDF5_LIBRARY_DIR}")
MESSAGE(STATUS "APP HDF5 libs: ${HDF5_LIBRARIES}")
MESSAGE(STATUS "APP HDF5 include dir: a ${HDF5_INCLUDE_DIR} b ${HDF5_INCLUDE_DIR_CPP}")
#MESSAGE(STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")
#MESSAGE(STATUS "HDF5_C_LIBRARIES: ${HDF5_C_LIBRARIES}")
#MESSAGE(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")

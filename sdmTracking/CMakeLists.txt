SET( SUBPROJECT_NAME sdmTracking )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

#OpenCV:
FIND_PACKAGE( OpenCV 2.4.3 REQUIRED core imgproc highgui objdetect features2d nonfree )
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

#find_path(VLFEAT_INCLUDE_DIR vl/sift.h HINTS "C:\\Users\\Patrik\\Documents\\GitHub\\vlfeat\\vlfeat-0.9.17")
#find_library(VLFEAT_LIBRARY NAMES vl HINTS "C:\\Users\\Patrik\\Documents\\GitHub\\vlfeat\\vlfeat-0.9.17\\bin\\win64")
#find_package_message(VLFEAT "VLFEAT: ${VLFEAT_LIBRARY}" "[${VLFEAT_LIBRARY}][${VLFEAT_INCLUDE_DIR}]")
#MESSAGE(STATUS "VLFEAT ${VLFEAT_INCLUDE_DIR}")
#MESSAGE(STATUS "VLFEAT ${VLFEAT_LIBRARY}")

#Source and header files:
SET(SOURCE
	sdmTracking.cpp
	hog.c
)

SET(HEADERS
	hog.h
)

ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

INCLUDE_DIRECTORIES( ${MATLAB_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
#include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${SupervisedDescentModel_SOURCE_DIR}/include )

#include_directories( ${VLFEAT_INCLUDE_DIR} )

#Make the app depend on the libraries
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} SupervisedDescentModel ImageIO ${KINECT_LIBNAME} Logging ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${OpenCV_LIBS} ${HDF5_LIBRARIES} ) #${VLFEAT_LIBRARY}

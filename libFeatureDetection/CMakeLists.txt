SET( SUBPROJECT_NAME libFeatureDetection )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "Configuring ${SUBPROJECT_NAME}:")
MESSAGE(STATUS "-------------------------------")

#OpenCV:
FIND_PACKAGE( OpenCV REQUIRED core imgproc highgui)
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
FIND_PACKAGE(Boost COMPONENTS random REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

#HDF5: (needed for reading the Model)
MESSAGE(STATUS "HDF5_ROOT: ${HDF5_ROOT}")
SET( CMAKE_MODULE_PATH ${HDF5_ROOT}/cmake/hdf5 )
MESSAGE(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
FIND_PACKAGE( HDF5 REQUIRED COMPONENTS C CXX ) # Maybe: FIND_PACKAGE (HDF5 REQURIED), and then, further up, PROJECT (HDF5MyApp C CXX) ? (according to hdf5 docu)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
MESSAGE(STATUS "HDF5 include dir: ${HDF5_INCLUDE_DIRS}")
MESSAGE(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")

#Source and header files:
SET(SOURCE
	CascadeERT.cpp
	CascadeFacialFeaturePoints.cpp
	CascadeFacialFeaturePointsSimple.cpp
	CascadeWvmOeSvmOe.cpp
	CascadeWvmSvm.cpp
	CircleDetector.cpp
	DetectorSVM.cpp
	DetectorWVM.cpp
	FdImage.cpp
	FdPatch.cpp
	FdPoint.cpp
	IImg.cpp
	MatlabReader.cpp
	OverlapElimination.cpp
	Pyramid.cpp
	Rect.cpp
	RegressorSVR.cpp
	RegressorWVR.cpp
	SkinDetector.cpp
	SLogger.cpp
	stdafx.cpp
	StdImage.cpp
	VDetector.cpp
	VDetectorVectorMachine.cpp
	FeaturePointsModelRANSAC.cpp
)

SET(HEADERS
	CascadeERT.h
	CascadeFacialFeaturePoints.h
	CascadeFacialFeaturePointsSimple.h
	CascadeWvmOeSvmOe.h
	CascadeWvmSvm.h
	CircleDetector.h
	DetectorSVM.h
	DetectorWVM.h
	FdImage.h
	FdPatch.h
	FdPoint.h
	IImg.h
	MatlabReader.h
	OverlapElimination.h
	Pyramid.h
	Rect.h
	RegressorSVR.h
	RegressorWVR.h
	SkinDetector.h
	SLogger.h
	stdafx.h
	StdImage.h
	targetver.h
	VDetector.h
	VDetectorVectorMachine.h
	FeaturePointsModelRANSAC.h
)

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

MESSAGE(STATUS "Finished configuring ${SUBPROJECT_NAME}")
MESSAGE(STATUS "===============================")

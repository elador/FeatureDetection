SET( SUBPROJECT_NAME libFeatureDetection )	# my own variable, not cmake's
#CMAKE_MINIMUM_REQUIRED( VERSION 2.8 ) # dont set in non-project cmake files
PROJECT( ${SUBPROJECT_NAME} )
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

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
	VCascade.cpp
	VDetector.cpp
	VDetectorVectorMachine.cpp
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
	utility.h
	VCascade.h
	VDetector.h
	VDetectorVectorMachine.h
)

#Make the library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

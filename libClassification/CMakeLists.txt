SET( SUBPROJECT_NAME Classification ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)
#Matlab:
SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
FIND_PACKAGE( MATLAB REQUIRED )
IF(MATLAB_FOUND)
  MESSAGE(STATUS "Matlab found at ${MATLAB_INCLUDE_DIR}")
ELSE(MATLAB_FOUND)
  MESSAGE(FATAL_ERROR "Matlab not found")
ENDIF(MATLAB_FOUND)


# source and header files
SET(HEADERS
#	include/classification/ApproximateSigmoidParameterComputation.h
#	include/classification/FastLibSvmTraining.h
#	include/classification/FixedApproximateSigmoidParameterComputation.h
#	include/classification/FixedSizeLibSvmTraining.h
#	include/classification/FrameBasedLibSvmTraining.h
#	include/classification/LibSvmClassifier.h
#	include/classification/LibSvmParameterBuilder.h
#	include/classification/LibSvmTraining.h
#	include/classification/PolyLibSvmParameterBuilder.h
#	include/classification/RbfLibSvmParameterBuilder.h
#	include/classification/RegressorSVR.h
#	include/classification/RegressorWVR.h
#	include/classification/SigmoidParameterComputation.h
#	include/classification/TrainableTwoStageClassifier.h
#	include/classification/TwoStageClassifier.h
	include/classification/WvmClassifier.hpp
	include/classification/BinaryClassifier.hpp
	include/classification/ProbabilisticClassifier.hpp
	include/classification/VectorMachineClassifier.hpp
	include/classification/SvmClassifier.hpp
#	include/classification/TrainableClassifier.h
#	include/classification/Training.h
	include/classification/ProbabilisticVectorMachineClassifier.hpp
	include/classification/Kernel.hpp
	include/classification/RbfKernel.hpp
	include/classification/PolynomialKernel.hpp
#	include/svm.h
)
SET(SOURCE
#	src/classification/ApproximateSigmoidParameterComputation.cpp
	src/classification/DetectorWVM.cpp
#	src/classification/FastLibSvmTraining.cpp
#	src/classification/FixedApproximateSigmoidParameterComputation.cpp
#	src/classification/FixedSizeLibSvmTraining.cpp
#	src/classification/FrameBasedLibSvmTraining.cpp
#	src/classification/LibSvmClassifier.cpp
#	src/classification/LibSvmParameterBuilder.cpp
#	src/classification/LibSvmTraining.cpp
#	src/classification/PolyLibSvmParameterBuilder.cpp
#	src/classification/RbfLibSvmParameterBuilder.cpp
#	src/classification/RegressorSVR.cpp
#	src/classification/RegressorWVR.cpp
#	src/classification/SigmoidParameterComputation.cpp
#	src/classification/TrainableTwoStageClassifier.cpp
#	src/classification/TwoStageClassifier.cpp
	src/classification/WvmClassifier.cpp
	src/classification/SvmClassifier.cpp
	src/classification/VectorMachineClassifier.cpp
	src/classification/ProbabilisticVectorMachineClassifier.cpp
	src/classification/RbfKernel.cpp
#	src/svm.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( ${MATLAB_INCLUDE_DIR} )
#include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

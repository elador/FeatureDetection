SET( SUBPROJECT_NAME Classification ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)

if(WITH_MATLAB_CLASSIFIER)
	set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
	find_package(MATLAB REQUIRED)
	if(MATLAB_FOUND)
		message(STATUS "Matlab found at ${MATLAB_INCLUDE_DIR}")
	else(MATLAB_FOUND)
		message(FATAL_ERROR "Matlab not found")
	endif(MATLAB_FOUND)
	include_directories(${MATLAB_INCLUDE_DIR})
	set(MATLAB_LIBNAME ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY})
	add_definitions(-DWITH_MATLAB_CLASSIFIER)
else()
	set(MATLAB_LIBNAME "")
endif()

# source and header files
SET(HEADERS
	include/classification/AgeBasedExampleManagement.hpp
	include/classification/BinaryClassifier.hpp
	include/classification/ConfidenceBasedExampleManagement.hpp
	include/classification/EmptyExampleManagement.hpp
	include/classification/ExampleManagement.hpp
	include/classification/FixedTrainableProbabilisticSvmClassifier.hpp
	include/classification/FrameBasedExampleManagement.hpp
	include/classification/HistogramIntersectionKernel.hpp
	include/classification/IImg.hpp
	include/classification/Kernel.hpp
	include/classification/KernelVisitor.hpp
	include/classification/LinearKernel.hpp
	include/classification/PolynomialKernel.hpp
	include/classification/ProbabilisticClassifier.hpp
	include/classification/ProbabilisticRvmClassifier.hpp
	include/classification/ProbabilisticSvmClassifier.hpp
	include/classification/ProbabilisticTwoStageClassifier.hpp
	include/classification/ProbabilisticWvmClassifier.hpp
	include/classification/RbfKernel.hpp
	include/classification/RvmClassifier.hpp
	include/classification/SvmClassifier.hpp
	include/classification/TrainableBinaryClassifier.hpp
	include/classification/TrainableClassifier.hpp
	include/classification/TrainableProbabilisticClassifier.hpp
	include/classification/TrainableProbabilisticSvmClassifier.hpp
	include/classification/TrainableProbabilisticTwoStageClassifier.hpp
	include/classification/TrainableSvmClassifier.hpp
	include/classification/UnlimitedExampleManagement.hpp
	include/classification/VectorBasedExampleManagement.hpp
	include/classification/VectorMachineClassifier.hpp
	include/classification/WvmClassifier.hpp
)
SET(SOURCE
	src/classification/AgeBasedExampleManagement.cpp
	src/classification/ConfidenceBasedExampleManagement.cpp
	src/classification/FrameBasedExampleManagement.cpp
	src/classification/IImg.cpp
	src/classification/ProbabilisticRvmClassifier.cpp
	src/classification/ProbabilisticSvmClassifier.cpp
	src/classification/ProbabilisticTwoStageClassifier.cpp
	src/classification/ProbabilisticWvmClassifier.cpp
	src/classification/RvmClassifier.cpp
	src/classification/SvmClassifier.cpp
	src/classification/TrainableProbabilisticSvmClassifier.cpp
	src/classification/TrainableSvmClassifier.cpp
	src/classification/UnlimitedExampleManagement.cpp
	src/classification/VectorBasedExampleManagement.cpp
	src/classification/VectorMachineClassifier.cpp
	src/classification/WvmClassifier.cpp
)

include_directories("include")

# add dependencies
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} ${MATLAB_LIBNAME})

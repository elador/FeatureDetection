SET( SUBPROJECT_NAME Classification ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)

# source and header files
SET(HEADERS
	include/classification/AgeBasedExampleManagement.hpp
	include/classification/BinaryClassifier.hpp
	include/classification/ConfidenceBasedExampleManagement.hpp
	include/classification/EmptyExampleManagement.hpp
	include/classification/ExampleManagement.hpp
	include/classification/FixedTrainableProbabilisticSvmClassifier.hpp
	include/classification/Kernel.hpp
	include/classification/KernelVisitor.hpp
	include/classification/LinearKernel.hpp
	include/classification/PolynomialKernel.hpp
	include/classification/ProbabilisticClassifier.hpp
	include/classification/ProbabilisticSvmClassifier.hpp
	include/classification/RbfKernel.hpp
	include/classification/SvmClassifier.hpp
	include/classification/TrainableBinaryClassifier.hpp
	include/classification/TrainableClassifier.hpp
	include/classification/TrainableProbabilisticClassifier.hpp
	include/classification/TrainableProbabilisticSvmClassifier.hpp
	include/classification/TrainableSvmClassifier.hpp
	include/classification/UnlimitedExampleManagement.hpp
	include/classification/VectorBasedExampleManagement.hpp
	include/classification/VectorMachineClassifier.hpp
)
SET(SOURCE
	src/classification/AgeBasedExampleManagement.cpp
	src/classification/ConfidenceBasedExampleManagement.cpp
	src/classification/ProbabilisticSvmClassifier.cpp
	src/classification/SvmClassifier.cpp
	src/classification/TrainableProbabilisticSvmClassifier.cpp
	src/classification/TrainableSvmClassifier.cpp
	src/classification/UnlimitedExampleManagement.cpp
	src/classification/VectorBasedExampleManagement.cpp
	src/classification/VectorMachineClassifier.cpp
)

include_directories("include")

# add dependencies
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ${OpenCV_LIBS})

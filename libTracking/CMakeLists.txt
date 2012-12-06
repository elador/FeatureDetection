SET( SUBPROJECT_NAME Tracking ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/svm.h
	include/classification/ApproximateSigmoidParameterComputation.h
	include/classification/Classifier.h
	include/classification/FastSvmTraining.h
	include/classification/FeatureExtractor.h
	include/classification/FeatureVector.h
	include/classification/FixedApproximateSigmoidParameterComputation.h
	include/classification/FrameBasedSvmTraining.h
	include/classification/HistEqFeatureExtractor.h
	include/classification/ImagePyramidFeatureExtractor.h
	include/classification/LibSvmClassifier.h
	include/classification/LibSvmParameterBuilder.h
	include/classification/LibSvmTraining.h
	include/classification/PolyLibSvmParameterBuilder.h
	include/classification/RbfLibSvmParameterBuilder.h
	include/classification/SigmoidParameterComputation.h
	include/classification/TrainableClassifier.h
	include/classification/Training.h
	include/tracking/AdaptiveCondensationTracker.h
	include/tracking/AdaptiveMeasurementModel.h
	include/tracking/CondensationTracker.h
	include/tracking/DensityPositionExtractor.h
	include/tracking/DualSvmModel.h
	include/tracking/FilteringPositionExtractor.h
	include/tracking/GridSampler.h
	include/tracking/LowVarianceSampling.h
	include/tracking/MaxWeightPositionExtractor.h
	include/tracking/MeasurementModel.h
	include/tracking/PatchDuplicateFilter.h
	include/tracking/PositionDependentMeasurementModel.h
	include/tracking/PositionExtractor.h
	include/tracking/Rectangle.h
	include/tracking/ResamplingAlgorithm.h
	include/tracking/ResamplingSampler.h
	include/tracking/Sample.h
	include/tracking/Sampler.h
	include/tracking/SelfLearningMeasurementModel.h
	include/tracking/SimpleTransitionModel.h
	include/tracking/TransitionModel.h
	include/tracking/WeightedMeanPositionExtractor.h
	include/tracking/WvmSvmModel.h
)
SET(SOURCE
	src/svm.cpp
	src/classification/ApproximateSigmoidParameterComputation.cpp
	src/classification/FastSvmTraining.cpp
	src/classification/FeatureVector.cpp
	src/classification/FixedApproximateSigmoidParameterComputation.cpp
	src/classification/FrameBasedSvmTraining.cpp
	src/classification/HistEqFeatureExtractor.cpp
	src/classification/ImagePyramidFeatureExtractor.cpp
	src/classification/LibSvmClassifier.cpp
	src/classification/LibSvmParameterBuilder.cpp
	src/classification/LibSvmTraining.cpp
	src/classification/PolyLibSvmParameterBuilder.cpp
	src/classification/RbfLibSvmParameterBuilder.cpp
	src/classification/SigmoidParameterComputation.cpp
	src/tracking/AdaptiveCondensationTracker.cpp
	src/tracking/CondensationTracker.cpp
	src/tracking/DensityPositionExtractor.cpp
	src/tracking/DualSvmModel.cpp
	src/tracking/FilteringPositionExtractor.cpp
	src/tracking/GridSampler.cpp
	src/tracking/LowVarianceSampling.cpp
	src/tracking/MaxWeightPositionExtractor.cpp
	src/tracking/PatchDuplicateFilter.cpp
	src/tracking/PositionDependentMeasurementModel.cpp
	src/tracking/Rectangle.cpp
	src/tracking/ResamplingSampler.cpp
	src/tracking/Sample.cpp
	src/tracking/SelfLearningMeasurementModel.cpp
	src/tracking/SimpleTransitionModel.cpp
	src/tracking/WeightedMeanPositionExtractor.cpp
	src/tracking/WvmSvmModel.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

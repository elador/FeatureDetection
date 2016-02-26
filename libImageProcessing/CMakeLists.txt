SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(Boost 1.48.0 REQUIRED) # We only need the headers

FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/imageprocessing/BinningFilter.hpp
	include/imageprocessing/CachingFeatureExtractor.hpp
	include/imageprocessing/CachingPyramidFeatureExtractor.hpp
	include/imageprocessing/CellBasedPyramidFeatureExtractor.hpp
	include/imageprocessing/ChainedFilter.hpp
	include/imageprocessing/ColorBinningFilter.hpp
	include/imageprocessing/ColorChannelFilter.hpp
	include/imageprocessing/ColorSpaceConversionFilter.hpp
	include/imageprocessing/CompleteExtendedHogFilter.hpp
	include/imageprocessing/ConversionFilter.hpp
	include/imageprocessing/ConvolutionFilter.hpp
	include/imageprocessing/DirectImageFeatureExtractor.hpp
	include/imageprocessing/DirectPyramidFeatureExtractor.hpp
	include/imageprocessing/ExtendedHogFeatureExtractor.hpp
	include/imageprocessing/ExtendedHogFilter.hpp
	include/imageprocessing/FeatureExtractor.hpp
	include/imageprocessing/FilteringFeatureExtractor.hpp
	include/imageprocessing/FilteringPyramidFeatureExtractor.hpp
	include/imageprocessing/GammaCorrectionFilter.hpp
	include/imageprocessing/GradientBinningFilter.hpp
	include/imageprocessing/GradientChannelFilter.hpp
	include/imageprocessing/GradientFilter.hpp
	include/imageprocessing/GradientMagnitudeFilter.hpp
	include/imageprocessing/GradientSumFilter.hpp
	include/imageprocessing/GrayscaleFilter.hpp
	include/imageprocessing/GreyWorldNormalizationFilter.hpp
	include/imageprocessing/HaarFeatureFilter.hpp
	include/imageprocessing/HistEq64Filter.hpp
	include/imageprocessing/HistogramEqualizationFilter.hpp
	include/imageprocessing/HistogramFilter.hpp
	include/imageprocessing/HogFilter.hpp
	include/imageprocessing/ImageFilter.hpp
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/ImagePyramidLayer.hpp
	include/imageprocessing/IntegralChannelFeatureFilter.hpp
	include/imageprocessing/IntegralChannelHistogramFilter.hpp
	include/imageprocessing/IntegralFeatureExtractor.hpp
	include/imageprocessing/IntegralGradientFilter.hpp
	include/imageprocessing/IntegralImageFilter.hpp
	include/imageprocessing/LbpFilter.hpp
	include/imageprocessing/ParallelFilter.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/PatchResizingFeatureExtractor.hpp
	include/imageprocessing/PyramidFeatureExtractor.hpp
	include/imageprocessing/PyramidHogFilter.hpp
	include/imageprocessing/ReshapingFilter.hpp
	include/imageprocessing/ResizingFilter.hpp
	include/imageprocessing/SpatialHistogramFilter.hpp
	include/imageprocessing/SpatialPyramidHistogramFilter.hpp
	include/imageprocessing/UnitNormFilter.hpp
	include/imageprocessing/Version.hpp
	include/imageprocessing/VersionedImage.hpp
	include/imageprocessing/WhiteningFilter.hpp
	include/imageprocessing/ZeroMeanUnitVarianceFilter.hpp
	include/imageprocessing/extraction/AggregatedFeaturesExtractor.hpp
	include/imageprocessing/filtering/AggregationFilter.hpp
	include/imageprocessing/filtering/BgrToLuvConverter.hpp
	include/imageprocessing/filtering/BoxConvolutionFilter.hpp
	include/imageprocessing/filtering/ChannelExtractor.hpp
	include/imageprocessing/filtering/FhogAggregationFilter.hpp
	include/imageprocessing/filtering/FhogFilter.hpp
	include/imageprocessing/filtering/FilterTree.hpp
	include/imageprocessing/filtering/FpdwFeaturesFilter.hpp
	include/imageprocessing/filtering/GradientFilter.hpp
	include/imageprocessing/filtering/GradientHistogramFilter.hpp
	include/imageprocessing/filtering/GradientMagnitudeFilter.hpp
	include/imageprocessing/filtering/GradientOrientationFilter.hpp
	include/imageprocessing/filtering/HistogramFilter.hpp
	include/imageprocessing/filtering/TriangularConvolutionFilter.hpp
)
SET(SOURCE
	src/imageprocessing/CachingFeatureExtractor.cpp
	src/imageprocessing/CachingPyramidFeatureExtractor.cpp
	src/imageprocessing/CellBasedPyramidFeatureExtractor.cpp
	src/imageprocessing/ChainedFilter.cpp
	src/imageprocessing/ColorBinningFilter.cpp
	src/imageprocessing/ColorChannelFilter.cpp
	src/imageprocessing/ColorSpaceConversionFilter.cpp
	src/imageprocessing/CompleteExtendedHogFilter.cpp
	src/imageprocessing/ConversionFilter.cpp
	src/imageprocessing/ConvolutionFilter.cpp
	src/imageprocessing/DirectImageFeatureExtractor.cpp
	src/imageprocessing/DirectPyramidFeatureExtractor.cpp
	src/imageprocessing/ExtendedHogFeatureExtractor.cpp
	src/imageprocessing/ExtendedHogFilter.cpp
	src/imageprocessing/GammaCorrectionFilter.cpp
	src/imageprocessing/GradientBinningFilter.cpp
	src/imageprocessing/GradientChannelFilter.cpp
	src/imageprocessing/GradientFilter.cpp
	src/imageprocessing/GradientMagnitudeFilter.cpp
	src/imageprocessing/GradientSumFilter.cpp
	src/imageprocessing/GrayscaleFilter.cpp
	src/imageprocessing/GreyWorldNormalizationFilter.cpp
	src/imageprocessing/HaarFeatureFilter.cpp
	src/imageprocessing/HistEq64Filter.cpp
	src/imageprocessing/HistogramEqualizationFilter.cpp
	src/imageprocessing/HistogramFilter.cpp
	src/imageprocessing/HogFilter.cpp
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/IntegralChannelFeatureFilter.cpp
	src/imageprocessing/IntegralChannelHistogramFilter.cpp
	src/imageprocessing/IntegralGradientFilter.cpp
	src/imageprocessing/IntegralImageFilter.cpp
	src/imageprocessing/LbpFilter.cpp
	src/imageprocessing/ParallelFilter.cpp
	src/imageprocessing/PyramidHogFilter.cpp
	src/imageprocessing/ReshapingFilter.cpp
	src/imageprocessing/ResizingFilter.cpp
	src/imageprocessing/SpatialHistogramFilter.cpp
	src/imageprocessing/SpatialPyramidHistogramFilter.cpp
	src/imageprocessing/UnitNormFilter.cpp
	src/imageprocessing/Version.cpp
	src/imageprocessing/WhiteningFilter.cpp
	src/imageprocessing/ZeroMeanUnitVarianceFilter.cpp
	src/imageprocessing/extraction/AggregatedFeaturesExtractor.cpp
	src/imageprocessing/filtering/AggregationFilter.cpp
	src/imageprocessing/filtering/BgrToLuvConverter.cpp
	src/imageprocessing/filtering/BoxConvolutionFilter.cpp
	src/imageprocessing/filtering/ChannelExtractor.cpp
	src/imageprocessing/filtering/FhogAggregationFilter.cpp
	src/imageprocessing/filtering/FhogFilter.cpp
	src/imageprocessing/filtering/FilterTree.cpp
	src/imageprocessing/filtering/FpdwFeaturesFilter.cpp
	src/imageprocessing/filtering/GradientFilter.cpp
	src/imageprocessing/filtering/GradientHistogramFilter.cpp
	src/imageprocessing/filtering/GradientMagnitudeFilter.cpp
	src/imageprocessing/filtering/GradientOrientationFilter.cpp
	src/imageprocessing/filtering/HistogramFilter.cpp
	src/imageprocessing/filtering/TriangularConvolutionFilter.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
target_link_libraries(${SUBPROJECT_NAME} Logging ${Boost_LIBRARIES} ${OpenCV_LIBS})

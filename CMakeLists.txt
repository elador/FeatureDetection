set(PROJECT_NAME FeatureDetection)
project(${PROJECT_NAME})
cmake_minimum_required(VERSION 2.8)

# g++ needs a compiler flag to use the (experimental) C++11 support
# TODO: * We could test for VS/gcc/unsupported compilers and warn
#       * This check doesn't interfere with the compilation on windows, but is not necessary there.
#       * We should stop and throw an error if we're on gcc and no c++11 support is found
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_CXX11_FLAG)
if (HAS_CXX11_FLAG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# Core libraries:
add_subdirectory(libLogging)			# Text logging
add_subdirectory(libImageIO)			# Input-library for cameras, Kinect and hard-drive images.
add_subdirectory(libImageProcessing)	# Image pyramids, patch extraction, feature transformation
add_subdirectory(libClassification)		# Classification library
add_subdirectory(libSvm)				# libSVM (support vector classification)
add_subdirectory(libCondensation)		# Condensation tracking (particle filter)

# Tracking apps:
add_subdirectory(adaptiveTrackingApp)	# Adaptive tracking app.
add_subdirectory(trackingBenchmarkApp)	# Benchmark app for adaptive condensation tracking.
add_subdirectory(VideoPlayerApp)		# App for playing videos, showing annotations and saving videos with annotations.


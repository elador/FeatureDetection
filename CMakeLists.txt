SET( PROJECT_NAME FeatureDetection )
PROJECT( ${PROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# G++ needs a compiler flag to use the (experimental) C++11 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_CXX11_FLAG)
if (HAS_CXX11_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

#add_subdirectory(libImageIO)			# Input-library for cameras, Kinect and hard-drive images.
#add_subdirectory(libImageProcessing)	# Image pyramids, patch extraction, feature transformation
add_subdirectory(libClassification)		# Classification and regression library

add_subdirectory(mergePlaygroundApp)	# Test-app for the refactored/merged code

#add_subdirectory(libCondensation)		# Condensation tracking (particle filter)
#add_subdirectory(libDetection)			# Sliding-window detection, Skin detection
#add_subdirectory(libHeadDetection)		# Head/face specific things; 2D/3D models; Combining feature points
#add_subdirectory(libLogging)			# Text- and Image logging

#add_subdirectory(faceTrackingApp)		#
#add_subdirectory(headTrackingApp)		#
#add_subdirectory(ffpDetectApp)			# The classic MR-style face-detect app (4 stages, SvmOeWvmOe).
#add_subdirectory(faceDetectApp)			# The new, smart face- and facial feature point detection app.

#add_subdirectory(libRender)				# A tiny software renderer to render lists of 3D vertices
#add_subdirectory(renderTestApp)			# Test app for libRender

#add_subdirectory(trackerAppERT)		# kind of deprecated, was only a test. But can use this to see how to make a real-time tracker.
#add_subdirectory(patchDetectApp)		# deprecated, does not build anymore. Was used to classify 20x20 patches only.

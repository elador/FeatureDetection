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

# All the options for building the library. Can be changed on the command-line or in initial_cache.cmake.
message(STATUS "Options:")
option(WITH_MSKINECT_SDK "Enables the use of a Kinect as a webcam (using the MS Kinect SDK for Windows)" OFF)
message(STATUS "WITH_MSKINECT_SDK: ${WITH_MSKINECT_SDK}")
option(WITH_MATLAB_CLASSIFIER "Enables support for loading Matlab classifiers (e.g. WVMs) in libClassification and certain apps" OFF)
message(STATUS "WITH_MATLAB_CLASSIFIER: ${WITH_MATLAB_CLASSIFIER}")
option(WITH_LIBLINEAR_CLASSIFIER "Enables support for loading libLinear classifiers in certain apps (needs BLAS)" OFF)
message(STATUS "WITH_LIBLINEAR_CLASSIFIER: ${WITH_LIBLINEAR_CLASSIFIER}")

# Core libraries
add_subdirectory(libLogging)			# Text logging
add_subdirectory(libImageLogging)		# Image logging

add_subdirectory(libImageIO)			# Input-library for cameras, Kinect and hard-drive images.
add_subdirectory(libImageProcessing)	# Image pyramids, patch extraction, feature transformation
add_subdirectory(libClassification)		# Classification and regression library
add_subdirectory(libSvm)				# libSVM (support vector classification and regression)
if(WITH_LIBLINEAR_CLASSIFIER)
	add_subdirectory(libLinear)				# liblinear (linear classification with SVM and regression)
endif()

add_subdirectory(libDetection)			# Sliding-window detection, Skin detection (Note: Do we use it anymore? Yes, Condensation depends on it)
add_subdirectory(libCondensation)		# Condensation tracking (particle filter)

add_subdirectory(libRender)				# Mesh and Vertex classes. (And, currently not working so great: A software renderer to render vertices and 3D-meshes)
add_subdirectory(libMorphableModel)		# PCA- and 3D Morphable Face Model classes; Camera (Pose) estimation
add_subdirectory(libSupervisedDescentModel) # A non-parametric model for landmark detection. Uses the supervised descent method.

# Tracking apps
add_subdirectory(benchmarkApp)			# Benchmark app for feature extractors and classifiers in a tracking-like online learning scenario.
add_subdirectory(faceTrackingApp)		# Face tracking app (no adaptation to target).
add_subdirectory(adaptiveTrackingApp)	# Adaptive tracking app.
add_subdirectory(partiallyAdaptiveTrackingApp)	# Old adaptive tracking app.
add_subdirectory(headTrackingApp)		# Adaptive head tracking app.
add_subdirectory(VideoPlayerApp)		# App for playing videos, showing landmarks and storing videos.

# Face-detection apps:
#add_subdirectory(ffpDetectApp)			# The classic MR-style face-detect app (4 stages, SvmOeWvmOe). (Note: Check if it still works)

# SDM landmark detection apps
#add_subdirectory(sdmTraining)			# Training of Supervised Descent Method landmark detection models
#add_subdirectory(sdmLandmarkDetection)	# Run the SDM landmark detection on one or several images.
#add_subdirectory(sdmLandmarkDetectionPlain)	# Run the SDM landmark detection on one or several images. Version without libImageIO.
#add_subdirectory(sdmEvaluation)			# App to evaluate an SDM landmark model on images, given ground-truth landmarks

# 3DMM fitting
#add_subdirectory(fitter)			# Experimental app (compiles)

# Tools
#add_subdirectory(landmarkVisualiser)	# Simple app to read landmarks and images and display them
#add_subdirectory(landmarkConverter)		# Simple app to convert landmarks from one format into another

# Face-recognition
#add_subdirectory(frRunCogExp)		# App for doing face recognition tests with a gallery and probe images
#add_subdirectory(frDbTool)		# App for managing (creating and populating) the sqlite-database used for face-recognition experiments
#add_subdirectory(frDbExp)		# App for doing experiments with the face recognition scores in the database

# ============================================================================
# Experimental, non-working or deprecated
#add_subdirectory(landmarkEvaluation)	# Simple app to read detected landmarks and groundtruth (and an optional config to specify which landmarks to compare) and writes the differences to a file.
#add_subdirectory(faceDetectApp)			# The new, smart face- and facial feature point detection app. (Note: Check if we can delete it)
#add_subdirectory(facemodelTracking)		# Todo - doesn't compile atm
#add_subdirectory(sdmTracking)			# Experimental app TODO
#add_subdirectory(generateSyntheticLandmarkDataKlaus)	# An app specifically created for the task of generating synthetic landmark data from random faces from the 3DMM.
#add_subdirectory(3dmmRendererGUI)		# An interactive GUI for rendering 3D-meshes, and if it's a morphable model, modify the principal components
#add_subdirectory(ffpDetectAppExp)		# Experimental playground for FFD with the 3DMM/POSIT
#add_subdirectory(trackerAppERT)		# kind of deprecated, was only a test. But can use this to see how to make a real-time tracker.
#add_subdirectory(patchConverter)	# Converts patches (in .txt format) to other feature spaces (e.g. gray to WHI)
#add_subdirectory(libShapeModels)		# Kind of deprecated. Head/face specific things; 2D/3D models; Combining feature points

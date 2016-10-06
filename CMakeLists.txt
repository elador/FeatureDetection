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
option(WITH_RENDER_QOPENGL "Enables support for a QOpenGL-based renderer in addition to the software renderer and a few interop functions (requires Qt5)" OFF)
message(STATUS "WITH_RENDER_QOPENGL: ${WITH_RENDER_QOPENGL}")
option(WITH_MORPHABLEMODEL_HDF5 "Enables support for loading statismo .h5 Morphable Models (requires HDF5)" OFF)
message(STATUS "WITH_MORPHABLEMODEL_HDF5: ${WITH_MORPHABLEMODEL_HDF5}")
option(WITH_FITTING_LIBEIGEN "Uses libEigen for linear algebra operations in the fitting instead of OpenCV." OFF) # not used yet
message(STATUS "WITH_FITTING_LIBEIGEN: ${WITH_FITTING_LIBEIGEN}")


# Core libraries:
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
add_subdirectory(libMorphableModel)		# PCA- and 3D Morphable Face Model classes
add_subdirectory(libFitting)			# Camera (Pose) estimation and shape-fitting
add_subdirectory(libSupervisedDescent)	# A non-parametric model for landmark detection. Uses the supervised descent method.
add_subdirectory(libFaceRecognition)	# Utility library for face recognition experiments, to work with databases, face-record, etc.

# Tracking apps:
add_subdirectory(benchmarkApp)			# Benchmark app for feature extractors and classifiers in a tracking-like online learning scenario.
add_subdirectory(trackingBenchmarkApp)	# Benchmark app for adaptive condensation tracking.
add_subdirectory(faceTrackingApp)		# Face tracking app (no adaptation to target).
add_subdirectory(adaptiveTrackingApp)	# Adaptive tracking app.
add_subdirectory(partiallyAdaptiveTrackingApp)	# Old adaptive tracking app.
add_subdirectory(headTrackingApp)		# Adaptive head tracking app.
add_subdirectory(VideoPlayerApp)		# App for playing videos, showing landmarks and storing videos.

# Detector training apps:
add_subdirectory(DetectorTrainingApp)	# Training app for cell-based detectors (aggregated channel features).

# Face-detection apps:
#add_subdirectory(ffpDetectApp)			# The classic MR-style face-detect app (4 stages, SvmOeWvmOe). (Note: Check if it still works)

# Face- and landmark detection apps:
add_subdirectory(detect-and-correct-faces) # Detect faces and validate them using given ground-truth.
add_subdirectory(detect-landmarks)		# Run the SDM landmark detection on one or several images, the input is either images (V&J will be run) or images & faceboxes.
add_subdirectory(sdmSimpleLandmarkDetection)	# Run the SDM landmark detection on just one image. Version without libImageIO. (well, libSupervisedDescentModel has a dependency on libImageIO now, but we could easily get rid of that with a CMake option)
add_subdirectory(sdmTraining)			# Training of Supervised Descent Method landmark detection models

# 3DMM fitting:
add_subdirectory(fitter)			# Experimental command-line fitting app using the software renderer.
add_subdirectory(fittingRenderer)	# App to render fitting results
add_subdirectory(fitterGUI)			# Experimental Fitting-app using the Software Renderer (for one image)
add_subdirectory(fittingTools) # Tools for fitting and model-rendering (e.g. compare isomaps)

# Tools:
add_subdirectory(landmarkVisualiser)	# Simple app to read landmarks and images and display them
add_subdirectory(landmarkConverter)		# Simple app to convert landmarks from one format into another
add_subdirectory(evaluate-landmarks)	# Read detected and ground-truth landmarks and perform an evaluation.

# Face-recognition (does not work because of hardcoded dependencies on proprietary software):
#add_subdirectory(facerecognitionTools) # Tools (e.g. create probe/gallery image-lists)
#add_subdirectory(frRunCogExp)		# App for doing face recognition tests with a gallery and probe images
#add_subdirectory(frDbTool)		# App for managing (creating and populating) the sqlite-database used for face-recognition experiments
#add_subdirectory(frDbExp)		# App for doing experiments with the face recognition scores in the database

# ============================================================================
# Experimental, non-working or deprecated:
#add_subdirectory(sdmTracking)			# Experimental app that does tracking using an SDM model.
#add_subdirectory(generateSyntheticLandmarkDataKlaus)	# An app specifically created for the task of generating synthetic landmark data from random faces from the 3DMM.
#add_subdirectory(softwareRendererTestApp) # Contains a QOpenGL and SW-renderer and was used to compare both, including shape-fitting.
#add_subdirectory(3dmmRendererGUI)		# An interactive GUI for rendering 3D-meshes, and if it's a morphable model, modify the principal components
#add_subdirectory(ffpDetectAppExp)		# Old experimental playground for FFD with the 3DMM/POSIT. It's deprecated (uses old code from libShapeModels). Used to generate the results for the FDDB benchmark.
#add_subdirectory(patchConverter)	# Converts patches (in .txt format) to other feature spaces (e.g. gray to WHI)

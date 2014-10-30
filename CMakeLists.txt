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

#add_subdirectory(libDetection)			# Sliding-window detection, Skin detection (Note: Do we use it anymore? Yes, Condensation depends on it)
#add_subdirectory(libCondensation)		# Condensation tracking (particle filter)

add_subdirectory(libRender)				# Mesh and Vertex classes. (And, currently not working so great: A software renderer to render vertices and 3D-meshes)
add_subdirectory(libMorphableModel)		# PCA- and 3D Morphable Face Model classes
add_subdirectory(libFitting)			# Camera (Pose) estimation and shape-fitting
#add_subdirectory(libSupervisedDescent)	# A non-parametric model for landmark detection. Uses the supervised descent method.
add_subdirectory(libFaceRecognition)	# Utility library for face recognition experiments, to work with databases, face-record, etc.

# Tracking apps:
#add_subdirectory(benchmarkApp)			# Benchmark app for feature extractors and classifiers in a tracking-like online learning scenario.
#add_subdirectory(trackingBenchmarkApp)	# Benchmark app for adaptive condensation tracking.
#add_subdirectory(faceTrackingApp)		# Face tracking app (no adaptation to target).
#add_subdirectory(adaptiveTrackingApp)	# Adaptive tracking app.
#add_subdirectory(partiallyAdaptiveTrackingApp)	# Old adaptive tracking app.
#add_subdirectory(headTrackingApp)		# Adaptive head tracking app.
#add_subdirectory(VideoPlayerApp)		# App for playing videos, showing landmarks and storing videos.

# Face-detection apps:
#add_subdirectory(ffpDetectApp)			# The classic MR-style face-detect app (4 stages, SvmOeWvmOe). (Note: Check if it still works)

# Face- and landmark detection apps:
#add_subdirectory(detect-and-correct-faces) # Detect faces and validate them using given ground-truth.
#add_subdirectory(detect-landmarks)		# Run the SDM landmark detection on one or several images, the input is either images (V&J will be run) or images & faceboxes.
#add_subdirectory(sdmSimpleLandmarkDetection)	# Run the SDM landmark detection on just one image. Version without libImageIO. (well, libSupervisedDescentModel has a dependency on libImageIO now, but we could easily get rid of that with a CMake option)
#add_subdirectory(sdmTraining)			# Training of Supervised Descent Method landmark detection models

# 3DMM fitting:
add_subdirectory(fitter)			# Experimental command-line fitting app using the software renderer.
add_subdirectory(fittingRenderer)	# App to render fitting results
add_subdirectory(fitterGUI)			# Experimental Fitting-app using the Software Renderer (for one image)
add_subdirectory(compareIsomaps)	# Compare isomaps (extracted textures)

# Landmark tools:
#add_subdirectory(visualise-landmarks)	# Simple app to read landmarks and images and display them
#add_subdirectory(convert-landmarks)		# Simple app to convert landmarks from one format into another
#add_subdirectory(evaluate-landmarks)	# Read detected and ground-truth landmarks and perform an evaluation.

# Deep learning based frame selection:
add_subdirectory(train-frameselect-extract)	# Extract frames and label information for training. Random subset of vids & frames. With random selection, and (partly) negative patches. Also runs FR engine. Unfinished.
add_subdirectory(train-frameselect-extract-all)	# Input: Scores for every frame and the frames. Extracts frames and label information for training. All vids, all frames. Store as boost::serialization. Newest implementation.
add_subdirectory(train-frameselect-learn)	# Train tiny-cnn using the boost::serialization patches.
add_subdirectory(frames-labels-to-caffe)	# Convert boost::serialization file with patches and labels to Caffe format (currently: images. Later, I might add LevelDB/LMDB/HDF5)

# Simple frame selection:
add_subdirectory(frameselect-random)		# Extracts random frames from a video. Can be used as a baseline.
add_subdirectory(frameselect-simple)		# Rates frames with a simple weighting of blurriness, in-focus-ness, IED, head size.

# Face-recognition on PaSC:
add_subdirectory(convert-pasc-video-metadata-to-bs)	# Convert PaSC video detections and metadata (CSV) to boost::serialization format
add_subdirectory(crop-pasc-video-heads)		# Takes PaSC frames from our frameselection and the metadata and saves cropped head patches.
add_subdirectory(crop-pasc-video-heads-affinealign)	# Same as above, but does affine alignment / warping, then saves cropped head patches.
add_subdirectory(csv-framesfolder-to-pasc-csv) # Take a folder with frames and a CSV (converted from Matlab) with a matrix in the same order as the images in the folder. Then, use a PaSC XML sigset to generate a conforming CSV or BEE matrix file, with similarity-scores and the FTEs.

# Matching:
add_subdirectory(plot-video-scores)			# match-all-frames. Match each frame of PaSC videos, outputs the scores as boost::serialization file and to a CSV file to plot in Matlab
add_subdirectory(match-video)			# Matches a video sigset using fvsdk, writes out results to CSV. Very similar to plot-video-scores?
add_subdirectory(match-videos-single-frames) # Match the best frame of a video (pre-selected by a frameselection), output the full similarity matrix as CSV (BEE-ready)

# Tools for matching on PaSC, some a bit dated:
#add_subdirectory(convert-pasc-sigset)		# Convert a PaSC xml sigset to a text file in property tree format, using FaceRecords ==> Use it to read XML, but otherwise, I think I shouldn't use it anymore. (but we used the ptree format to create MultiPIE sigsets)
#add_subdirectory(pasc-video-matching)	# Runs the whole pipeline on the PaSC video-video experiment, including fitting. A bit outdated and unfinished - only the skeletton. But some useful comments!
# Tools for matching on MultiPIE, some a bit dated:
#add_subdirectory(generate-matchlist)		# Takes a gallery and probe sigset (ptree, FaceRecords) and a matching-config and generates a (huge...) matchlist in ptree format that can be used to run matching engines. For MultiPIE. Hmm, maybe not a sooo good idea?
#add_subdirectory(generate-sigset-multipie) # Specify subsets of MultiPIE and then gather the files and create FaceRecords and a ptree Sigset.
#add_subdirectory(generate-imglist-multipie)	# Specify subsets of MultiPIE and then gather the files and create an image list with paths to the images. Kind of similar to above. But MultiPIE has a non-trivial directory structure. How to handle it?
#add_subdirectory(frRunCogExp)		# Run FaceVACS executables enroll for probes & galleries. Commented out. New usage: Read 2 ptree sigsets and DataPathTransformation configs, enroll them using the SDK, match them, and store to the SQLite DB.
#add_subdirectory(frDbTool)		# Create and popule the sqlite-database used for face-recognition experiments
#add_subdirectory(frDbExp)		# Query the database for matching scores, store them as txt for plotting in Matlab.

# ============================================================================
# Experimental, non-working or deprecated:
#add_subdirectory(sdmTracking)			# Experimental app that does tracking using an SDM model.
#add_subdirectory(generateSyntheticLandmarkDataKlaus)	# An app specifically created for the task of generating synthetic landmark data from random faces from the 3DMM.
#add_subdirectory(softwareRendererTestApp) # Contains a QOpenGL and SW-renderer and was used to compare both, including shape-fitting.
#add_subdirectory(3dmmRendererGUI)		# An interactive GUI for rendering 3D-meshes, and if it's a morphable model, modify the principal components
#add_subdirectory(ffpDetectAppExp)		# Old experimental playground for FFD with the 3DMM/POSIT. It's deprecated (uses old code from libShapeModels). Used to generate the results for the FDDB benchmark.
#add_subdirectory(patchConverter)	# Converts patches (in .txt format) to other feature spaces (e.g. gray to WHI)

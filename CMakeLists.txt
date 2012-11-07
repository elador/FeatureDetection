SET( PROJECT_NAME FeatureDetection )
PROJECT( ${PROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
add_subdirectory(libImageIO)			# Input-library for cameras, Kinect and hard-drive images.
add_subdirectory(libFeatureDetection)	# Library for face feature detecion.
add_subdirectory(libTracking)
add_subdirectory(ffpDetectApp)			# The classic MR-style face-detect app (4 stages, SvmOeWvmOe).
add_subdirectory(faceTrackingApp)
add_subdirectory(faceDetectApp)			# The new, smart face- and facial feature point detection app.

#add_subdirectory(trackerAppERT)		# kind of deprecated, was only a test. But can use this to see how to make a real-time tracker.
#add_subdirectory(patchDetectApp)		# deprecated, does not build anymore. Was used to classify 20x20 patches only.

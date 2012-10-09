SET( PROJECT_NAME FeatureDetection )
PROJECT( ${PROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
add_subdirectory(libImageIO)
add_subdirectory(libFeatureDetection)
add_subdirectory(libKinectInterface)
add_subdirectory(libTracking)
add_subdirectory(ffpDetectApp)
add_subdirectory(faceTrackingApp)

add_subdirectory(kinectTestApp)

#add_subdirectory(trackerAppERT)	# kind of deprecated, was only a test. But can use this to see how to make a real-time tracker.
#add_subdirectory(patchDetectApp)	# deprecated, does not build anymore. Was used to classify 20x20 patches only.

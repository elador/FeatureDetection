#set(SUBPROJECT_NAME facerecognitionTools)
#project(${SUBPROJECT_NAME})

#message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

add_subdirectory(generateMultipieList) # Generate a list of MultiPIE images
add_subdirectory(generateMultipieSigset) # Generate lists of face recognition sets (sigsets) to be used as probe- and gallery sets (i.e. with subject metadata)

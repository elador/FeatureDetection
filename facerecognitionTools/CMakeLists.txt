#set(SUBPROJECT_NAME facerecognitionTools)
#project(${SUBPROJECT_NAME})

#message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

add_subdirectory(match)					#
add_subdirectory(enroll)				#

add_subdirectory(generateMatchlist)		# Takes a sigset and a matching-config and generates a matchlist that can be used to run matching engines.
add_subdirectory(generateMultipieList)	# Generate a list of MultiPIE images
add_subdirectory(generateMultipieSigset) # Generate lists of face recognition sets (sigsets) to be used as probe- and gallery sets (i.e. with subject metadata)
add_subdirectory(convertPascSigset)		# Convert a PaSC xml sigset to our text-format

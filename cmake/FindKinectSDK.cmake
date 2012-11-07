#SET(KinectSDK_ROOT_DIR "c:/Program Files/Microsoft SDKs/Kinect/v1.0/")
SET(KinectSDK_ROOT_DIR $ENV{KINECTSDK10_DIR})
message(${KinectSDK_ROOT_DIR})
SET(KinectSDK_LIB_DIR "${KinectSDK_ROOT_DIR}lib/x86")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)	# 64bit build
	SET(KinectSDK_LIB_DIR "${KinectSDK_ROOT_DIR}lib/amd64")
endif()
SET(KinectSDK_INCLUDE_DIR "${KinectSDK_ROOT_DIR}inc")
SET(KinectSDK_DLLS 
	"$ENV{SystemRoot}System32/Kinect10.dll"
)
SET(KinectSDK_LIBRARIES 
	"${KinectSDK_LIB_DIR}/Kinect10.lib"
)
SET(KinectSDK_INCLUDES 
	"${KinectSDK_INCLUDE_DIR}/NuiApi.h"
	"${KinectSDK_INCLUDE_DIR}/NuiImageCamera.h"
	"${KinectSDK_INCLUDE_DIR}/NuiSensor.h"
	"${KinectSDK_INCLUDE_DIR}/NuiSkeleton.h"
)

####################   Macro   #######################
MACRO(CHECK_FILES _FILES _DIR)
	SET(_MISSING_FILES)
	FOREACH(_FILE ${${_FILES}})
		IF(NOT EXISTS "${_FILE}")
			SET(KinectSDK_FOUND NO)
			get_filename_component(_FILE ${_FILE} NAME)
			SET(_MISSING_FILES "${_MISSING_FILES}${_FILE}, ")
		ENDIF()
	ENDFOREACH()
	IF(_MISSING_FILES)
		MESSAGE(STATUS "In folder \"${${_DIR}}\" not found files: ${_MISSING_FILES}")
		SET(KinectSDK_FOUND NO)
	ENDIF()
ENDMACRO(CHECK_FILES)

MACRO(CHECK_DIR _DIR)
	IF(NOT EXISTS "${${_DIR}}")
		MESSAGE(STATUS "Folder \"${${_DIR}}\" not found.")
		SET(KinectSDK_FOUND NO)
	ENDIF()
ENDMACRO(CHECK_DIR)

##################### Checking #######################
MESSAGE(STATUS "Searching for Microsoft KinectSDK...")
SET(KinectSDK_FOUND YES)

CHECK_DIR(KinectSDK_ROOT_DIR)
IF(KinectSDK_FOUND)
	CHECK_DIR(KinectSDK_LIB_DIR)
	CHECK_DIR(KinectSDK_INCLUDE_DIR)
	
	IF(KinectSDK_FOUND)
		#CHECK_FILES(KinectSDK_DLLS KinectSDK_ROOT_DIR)
		CHECK_FILES(KinectSDK_LIBRARIES KinectSDK_LIB_DIR)
		CHECK_FILES(KinectSDK_INCLUDES KinectSDK_INCLUDE_DIR)
	ENDIF()
ENDIF()

MESSAGE(STATUS "KinectSDK_FOUND - ${KinectSDK_FOUND}.")
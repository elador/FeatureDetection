set(SUBPROJECT_NAME FaceRecognition)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 COMPONENTS system filesystem serialization REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

find_package(OpenCV 2.4.3 REQUIRED core highgui)

# Add the FaceVACS SDK as a dependency:
set(FVSDKLIB_DBG "C:\\FVSDK_8_9_5\\lib\\x86_64\\msc_12.0-ipp_crtdll_g\\libfrsdk-8.9.5d")
set(FVSDKLIB "C:\\FVSDK_8_9_5\\lib\\x86_64\\msc_12.0-ipp_crtdll\\libfrsdk-8.9.5")
add_library(fvsdk SHARED IMPORTED)
set_property(TARGET fvsdk PROPERTY IMPORTED_LOCATION ${FVSDKLIB}.dll)
set_property(TARGET fvsdk PROPERTY IMPORTED_IMPLIB ${FVSDKLIB}.lib)
set_property(TARGET fvsdk PROPERTY IMPORTED_LOCATION_DEBUG ${FVSDKLIB_DBG}.dll)
set_property(TARGET fvsdk PROPERTY IMPORTED_IMPLIB_DEBUG ${FVSDKLIB_DBG}.lib)
include_directories("C:\\FVSDK_8_9_5\\include")

# source and header files
set(SOURCE
	src/facerecognition/FaceRecord.cpp
	src/facerecognition/pasc.cpp
	src/facerecognition/utils.cpp
)

set(HEADERS
	include/facerecognition/FaceRecord.hpp
	include/facerecognition/FaceVacsEngine.hpp
	include/facerecognition/frsdk.hpp
	include/facerecognition/pasc.hpp
	include/facerecognition/utils.hpp
)

include_directories("include")

# add dependencies
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${Logging_SOURCE_DIR}/include)

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ${OpenCV_LIBS} ${Boost_LIBRARIES} fvsdk) # ImageIO

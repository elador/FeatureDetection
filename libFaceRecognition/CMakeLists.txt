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
message(STATUS "FVSDK library at: ${FVSDK_LIB_DIR}")
if(WIN32) # I think this doesn't include Cygwin/MingW, but we would want it too
	message(STATUS "FVSDK debug library at: ${FVSDK_DBG_LIB_DIR}") # Windows only
	set(FVSDKLIB ${FVSDK_LIB_DIR})
	set(FVSDKLIB_DBG ${FVSDK_DBG_LIB_DIR})
	add_library(fvsdk SHARED IMPORTED)
	set_property(TARGET fvsdk PROPERTY IMPORTED_LOCATION ${FVSDKLIB}.dll)
	set_property(TARGET fvsdk PROPERTY IMPORTED_IMPLIB ${FVSDKLIB}.lib)
	set_property(TARGET fvsdk PROPERTY IMPORTED_LOCATION_DEBUG ${FVSDKLIB_DBG}.dll)
	set_property(TARGET fvsdk PROPERTY IMPORTED_IMPLIB_DEBUG ${FVSDKLIB_DBG}.lib)
else(WIN32)
	set(FVSDKLIB ${FVSDK_LIB_DIR})
	add_library(fvsdk SHARED IMPORTED)
	set_property(TARGET fvsdk PROPERTY IMPORTED_LOCATION ${FVSDKLIB}/libfrsdk-8.9.5.so)
endif()
message(STATUS "Including FVSDK at ${FVSDK_INCLUDE_DIR}")
include_directories(${FVSDK_INCLUDE_DIR})

# source and header files
set(SOURCE
	src/facerecognition/alignment.cpp
	src/facerecognition/EnrolCoutFeedback.cpp
	src/facerecognition/FaceRecord.cpp
	src/facerecognition/FaceVacsEngine.cpp
	src/facerecognition/frameassessment.cpp
	src/facerecognition/frsdk.cpp
	src/facerecognition/pasc.cpp
	src/facerecognition/utils.cpp
)

set(HEADERS
	include/facerecognition/alignment.hpp
	include/facerecognition/EnrolCoutFeedback.hpp
	include/facerecognition/FaceRecord.hpp
	include/facerecognition/FaceVacsEngine.hpp
	include/facerecognition/frameassessment.hpp
	include/facerecognition/frsdk.hpp
	include/facerecognition/pasc.hpp
	include/facerecognition/PathSerialization.hpp
	include/facerecognition/ThreadPool.hpp
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

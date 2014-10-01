set(SUBPROJECT_NAME train-frame-extract-nnet)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core imgproc)

find_package(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

set(TinyCNN_INCLUDE_DIR "C:\\Users\\Patrik\\Documents\\GitHub\\FeatureDetection\\3rdparty\\tiny-cnn\\include")
add_definitions(-DCNN_USE_AVX)
add_definitions(-DCNN_USE_SSE)

# Add the FaceVACS SDK as a dependency:
set(FVSDKLIB_DBG "C:\\FVSDK_8_9_5\\lib\\x86_64\\msc_12.0-ipp_crtdll_g\\libfrsdk-8.9.5d")
set(FVSDKLIB "C:\\FVSDK_8_9_5\\lib\\x86_64\\msc_12.0-ipp_crtdll\\libfrsdk-8.9.5")
add_library(fvsdk SHARED IMPORTED)
set_property(TARGET fvsdk PROPERTY IMPORTED_LOCATION ${FVSDKLIB}.dll)
set_property(TARGET fvsdk PROPERTY IMPORTED_IMPLIB ${FVSDKLIB}.lib)
set_property(TARGET fvsdk PROPERTY IMPORTED_LOCATION_DEBUG ${FVSDKLIB_DBG}.dll)
set_property(TARGET fvsdk PROPERTY IMPORTED_IMPLIB_DEBUG ${FVSDKLIB_DBG}.lib)
include_directories("C:\\FVSDK_8_9_5\\include")

# Source and header files:
set(SOURCE
	train-frame-extract-nnet.cpp
)

set(HEADERS
)

add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})

include_directories(${TinyCNN_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${FaceRecognition_SOURCE_DIR}/include)

# Make the app depend on the libraries
target_link_libraries(${SUBPROJECT_NAME} FaceRecognition ImageIO Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} fvsdk)

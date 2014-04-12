set(SUBPROJECT_NAME faceTrackingApp)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)

find_package(OpenCV 2.4.3 REQUIRED core highgui)

if(WITH_MSKINECT_SDK)
	# Include Microsoft Kinect SDK (Windows)
	set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
	find_package(KinectSDK REQUIRED)
	include_directories(${KinectSDK_INCLUDE_DIR})
	set(KINECT_LIBNAME ${KinectSDK_LIBRARIES})
	add_definitions(-DWITH_MSKINECT_SDK)
else()
	set(KINECT_LIBNAME "")
endif()

# source and header files
set(HEADERS
	FaceTracking.hpp
)
set(SOURCE
	FaceTracking.cpp
)

# add dependencies
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${SVM_SOURCE_DIR}/include)
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${Detection_SOURCE_DIR}/include)
include_directories(${Condensation_SOURCE_DIR}/include)

# make executable
add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Detection Condensation ImageIO ${KINECT_LIBNAME} ImageProcessing Classification SVM Logging ${Boost_LIBRARIES} ${OpenCV_LIBS})

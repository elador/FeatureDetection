set(SUBPROJECT_NAME cnn-matcher-learn)
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

set(TinyCNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/tiny-cnn/include")
#add_definitions(-DCNN_USE_AVX)
add_definitions(-DCNN_USE_SSE)
add_definitions(-DCNN_USE_TBB)
#add_definitions(-DCNN_TASK_SIZE 16) # 8 is default (in tiny-cnn/include/config.h). Which one takes precedence?

# Your-external "mylib"
add_library(tbb SHARED IMPORTED )
# You can define two import-locations: one for debug and one for release.
set_target_properties(tbb PROPERTIES IMPORTED_LOCATION /usr/lib/libtbb.so.2)


# Source and header files:
set(SOURCE
	cnn-matcher-learn.cpp
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
target_link_libraries(${SUBPROJECT_NAME} FaceRecognition ImageIO Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} tbb)

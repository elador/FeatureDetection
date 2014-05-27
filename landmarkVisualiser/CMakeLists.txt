set(SUBPROJECT_NAME landmarkVisualiser)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui calib3d)

find_package(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

# Source and header files:
set(SOURCE
	landmarkVisualiser.cpp
)

set(HEADERS
)

add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageLogging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)

target_link_libraries(${SUBPROJECT_NAME} ImageIO ImageLogging Logging ${Boost_LIBRARIES} ${OpenCV_LIBS})

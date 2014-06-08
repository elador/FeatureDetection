set(SUBPROJECT_NAME fittingRenderer)
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

# Todo: ifdef this or rather move to library anyway
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
include_directories(${EIGEN3_INCLUDE_DIR})

# Source and header files:
set(SOURCE
	fittingRenderer.cpp
)

set(HEADERS
)

add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${Render_SOURCE_DIR}/include)
include_directories(${MorphableModel_SOURCE_DIR}/include)
include_directories(${Fitting_SOURCE_DIR}/include)

# Make the app depend on the libraries
target_link_libraries(${SUBPROJECT_NAME} Fitting MorphableModel Render ImageIO Logging ${Boost_LIBRARIES} ${OpenCV_LIBS})

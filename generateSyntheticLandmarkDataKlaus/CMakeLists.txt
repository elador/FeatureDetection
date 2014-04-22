set(SUBPROJECT_NAME generateSyntheticLandmarkDataKlaus)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies:
find_package( OpenCV 2.4.3 REQUIRED core imgproc highgui)

find_package(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

#Source and header files:
set(SOURCE
	generateSyntheticLandmarkDataKlaus.cpp
)

set(HEADERS
)

include_directories(${Render_SOURCE_DIR}/include)
include_directories(${MorphableModel_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

#Make the library
add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Render MorphableModel ImageIO Logging ${OpenCV_LIBS} ${Boost_LIBRARIES})

set(SUBPROJECT_NAME detect-and-correct-faces)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# Find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui objdetect)

find_package(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

# Source and header files:
set(SOURCE
	detect-and-correct-faces.cpp
)

set(HEADERS
)

add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)

# Make the app depend on the libraries
target_link_libraries(${SUBPROJECT_NAME} ImageIO Logging ${Boost_LIBRARIES} ${OpenCV_LIBS})

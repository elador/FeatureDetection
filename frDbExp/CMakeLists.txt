set(SUBPROJECT_NAME frDbExp)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8.11)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)

find_package(Boost 1.48.0 COMPONENTS program_options filesystem system REQUIRED) # others (e.g.): system filesystem
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

set(CMAKE_AUTOMOC ON)
find_package(Qt5Sql)

#Source and header files:
set(SOURCE
	frDbExp.cpp
)

set(HEADERS
	frDbExp.hpp
)

#Make the library
add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${ImageProcessing_SOURCE_DIR}/include)
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${Detection_SOURCE_DIR}/include)

#Make the app depend on the libraries
target_link_libraries(${SUBPROJECT_NAME} Detection ImageIO ImageProcessing Classification Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} Qt5::Sql)

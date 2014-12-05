set(SUBPROJECT_NAME SupervisedDescent)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# Find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core imgproc features2d nonfree)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

find_package(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# source and header files
set(HEADERS
	include/superviseddescent/SdmLandmarkModel.hpp
	include/superviseddescent/superviseddescent.hpp
	include/superviseddescent/matserialisation.hpp
	include/superviseddescent/LandmarkBasedSupervisedDescentTraining.hpp
	include/superviseddescent/DescriptorExtractor.hpp
	include/superviseddescent/hog.h
	include/superviseddescent/OpenCVSiftFilter.hpp
	include/superviseddescent/VlHogFilter.hpp
	include/superviseddescent/utils.hpp
)
set(SOURCE
	src/superviseddescent/SdmLandmarkModel.cpp
	src/superviseddescent/LandmarkBasedSupervisedDescentTraining.cpp
	src/superviseddescent/hog.c
	src/superviseddescent/OpenCVSiftFilter.cpp
	src/superviseddescent/VlHogFilter.cpp
	src/superviseddescent/utils.cpp
)

include_directories("include")

# Add header includes:
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${ImageProcessing_SOURCE_DIR}/include)

# Make the library:
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} ImageProcessing ImageIO Logging ${OpenCV_LIBS} ${Boost_LIBRARIES})

# The install target:
install(TARGETS ${SUBPROJECT_NAME} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${SUBPROJECT_NAME}) # Todo: This is uppercase. Rename project to lowercase? This then clashes with toplevel project

# Unit tests:
# Set properties / executable of the test:
include_directories(${gtest_SOURCE_DIR}/include)

set(TEST_LINREG1D test_LinearRegressor1D)
add_executable(${TEST_LINREG1D} test/test_LinearRegressor1D.cpp)
target_link_libraries(${TEST_LINREG1D} ${SUBPROJECT_NAME} gtest_main gtest)
add_test(${TEST_LINREG1D} ${TEST_LINREG1D}) # testname exename

set(TEST_LINREGND test_LinearRegressorND)
add_executable(${TEST_LINREGND} test/test_LinearRegressorND.cpp)
target_link_libraries(${TEST_LINREGND} ${SUBPROJECT_NAME} gtest_main gtest)
add_test(${TEST_LINREGND} ${TEST_LINREGND})

set(TEST_SDO test_SupervisedDescentOptimiser)
add_executable(${TEST_SDO} test/test_SupervisedDescentOptimiser.cpp)
target_link_libraries(${TEST_SDO} ${SUBPROJECT_NAME} gtest_main gtest)
add_test(${TEST_SDO} ${TEST_SDO})

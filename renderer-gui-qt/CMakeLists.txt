set(SUBPROJECT_NAME renderer-gui-qt)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8.11)
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

set(CMAKE_AUTOMOC ON)
find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5Qml)
find_package(Qt5Quick)

set(QT_RESOURCES
	#qml/res.qrc
	window.qrc
	shared/shared.qrc
)
QT5_ADD_RESOURCES(RSCS ${QT_RESOURCES})

# Source and header files:
set(SOURCE
	renderer-gui-qt.cpp
	main.cpp
)

set(HEADERS
)

add_executable(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} ${RSCS})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Logging_SOURCE_DIR}/include)

target_link_libraries(${SUBPROJECT_NAME} Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} Qt5::Core Qt5::Gui Qt5::Qml Qt5::Quick)

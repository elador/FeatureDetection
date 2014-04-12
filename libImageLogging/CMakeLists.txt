SET( SUBPROJECT_NAME ImageLogging ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

FIND_PACKAGE(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

# source and header files
SET(HEADERS
	include/imagelogging/ImageLogger.hpp
	include/imagelogging/ImageLoggerFactory.hpp
	include/imagelogging/Appender.hpp
	include/imagelogging/ConsoleAppender.hpp
	include/imagelogging/ImageFileWriter.hpp
	include/imagelogging/imageloglevels.hpp
	include/imagelogging/ImageLoggerFunctionsFromOldLogger.hpp
)
SET(SOURCE
	src/imagelogging/ImageLogger.cpp
	src/imagelogging/ImageLoggerFactory.cpp
	src/imagelogging/ConsoleAppender.cpp
	src/imagelogging/ImageFileWriter.cpp
	src/imagelogging/ImageLoggerFunctionsFromOldLogger.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
target_link_libraries(${SUBPROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBS})

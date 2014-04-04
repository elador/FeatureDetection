SET( SUBPROJECT_NAME Logging ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies

# source and header files
SET(HEADERS
	include/logging/Logger.hpp
	include/logging/LoggerFactory.hpp
	include/logging/Appender.hpp
	include/logging/ConsoleAppender.hpp
	include/logging/FileAppender.hpp
	include/logging/LogLevels.hpp
)
SET(SOURCE
	src/logging/Logger.cpp
	src/logging/LoggerFactory.cpp
	src/logging/ConsoleAppender.cpp
	src/logging/FileAppender.cpp
)

include_directories( "include" )

# add dependencies

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

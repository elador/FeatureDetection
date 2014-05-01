set(SUBPROJECT_NAME Logging)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies

# source and header files
set(HEADERS
	include/logging/Logger.hpp
	include/logging/LoggerFactory.hpp
	include/logging/Appender.hpp
	include/logging/ConsoleAppender.hpp
	include/logging/FileAppender.hpp
	include/logging/LogLevels.hpp
)
set(SOURCE
	src/logging/Logger.cpp
	src/logging/LoggerFactory.cpp
	src/logging/ConsoleAppender.cpp
	src/logging/FileAppender.cpp
)

include_directories("include")

# add dependencies

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})

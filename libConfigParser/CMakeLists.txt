SET( SUBPROJECT_NAME ConfigParser ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
#Boost:
FIND_PACKAGE(Boost REQUIRED) # we need algorithm
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()


# source and header files
SET(HEADERS
	include/configparser/ConfigParser.hpp
)
SET(SOURCE
	src/configparser/ConfigParser.cpp
)

include_directories( "include" )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# add dependencies

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

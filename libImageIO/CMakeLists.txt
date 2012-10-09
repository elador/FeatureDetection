SET( SUBPROJECT_NAME ImageIO ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED filesystem)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/imageio/ImageSource.h
	include/imageio/DirectoryImageSource.h
	include/imageio/VideoImageSource.h
)
SET(SOURCE
	src/imageio/DirectoryImageSource.cpp
	src/imageio/VideoImageSource.cpp
)

include_directories( "include" )

# make library
add_library( ${SUBPROJECT_NAME} SHARED ${SOURCE} ${HEADERS} )


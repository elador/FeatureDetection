set(SUBPROJECT_NAME MorphableModel)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8.11) # we can change this back as soon as we don't need libRender anymore (or libRender doesn't need Qt/OpenGL anymore)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)

find_package(OpenCV 2.4.3 REQUIRED core calib3d)

if(WITH_MORPHABLEMODEL_HDF5)
	if(WIN32)
		find_package(HDF5 REQUIRED COMPONENTS C CXX NAMES hdf5) # COMPONENTS doesn't seem to work. NAMES puts it in Config-mode
		set (HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR}) # According to hdf5 cmake file, should use *_DIRS, but they don't define it in hdf5-config.cmake. So if we use the Config-mode with NAMES, it isn't set, but it is, when used as in linux.
	else()
		#find_package (HDF5 REQUIRED CONFIG)
		find_package(HDF5 REQUIRED COMPONENTS C CXX) # works on windows as well but links to dll's instead of lib's...
	endif()
	message(STATUS "HDF5 includes at: ${HDF5_INCLUDE_DIRS}")
	message(STATUS "HDF5 libraries at: ${HDF5_LIBRARIES}")
	include_directories(${HDF5_INCLUDE_DIRS})
	add_definitions(-DWITH_MORPHABLEMODEL_HDF5)
endif()

# source and header files
set(HEADERS
	include/morphablemodel/PcaModel.hpp
	include/morphablemodel/MorphableModel.hpp
	include/morphablemodel/OpenCVCameraEstimation.hpp
	include/morphablemodel/AffineCameraEstimation.hpp
	include/morphablemodel/Hdf5Utils.hpp
)
set(SOURCE
	src/morphablemodel/PcaModel.cpp
	src/morphablemodel/MorphableModel.cpp
	src/morphablemodel/OpenCVCameraEstimation.cpp
	src/morphablemodel/AffineCameraEstimation.cpp
	src/morphablemodel/Hdf5Utils.cpp
)

include_directories("include")

# add dependencies
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Render_SOURCE_DIR}/include)
include_directories(${ImageIO_SOURCE_DIR}/include)
include_directories(${Logging_SOURCE_DIR}/include)

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Render Logging ${OpenCV_LIBS} ${Boost_LIBRARIES} ${HDF5_LIBRARIES})

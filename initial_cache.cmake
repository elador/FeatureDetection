#set(GRAVIS_PREFIX             $ENV{GRAVIS_PREFIX}   CACHE STRING "The basis of the gravis installation")
#set(GRAVIS_PREFIX "/home/huber/gravis" CACHE STRING "The basis of the gravis installation")
#set(GRAVIS_INSTALL_PREFIX     ${GRAVIS_PREFIX}      CACHE STRING "where to install"       )
#set(BUILD_SHARED_LIBS         TRUE                  CACHE STRING "build shared libs"      )

#set(CMAKE_CXX_FLAGS_RELEASE           "-O3 -march=native -UDEBUG -DNDEBUG -fopenmp -Wall -Wno-array-bounds -Wunknown-pragmas"        CACHE STRING "CXX Flags used for release compilation mode")
#set(CMAKE_CXX_FLAGS_DEBUG             "-O0 -ggdb -march=native -UNDEBUG -DDEBUG -fopenmp -Wall -Wno-array-bounds -Wunknown-pragmas"          CACHE STRING "CXX Flags used for debug compilation mode")
#set(CMAKE_C_FLAGS_RELEASE             "-O3 -march=native -UDEBUG -DNDEBUG -fopenmp -Wall -Wno-array-bounds -Wunknown-pragmas"          CACHE STRING "C Flags used for release compilation mode")
#set(CMAKE_C_FLAGS_DEBUG               "-O0 -ggdb -march=native -UNDEBUG -DDEBUG -fopenmp -Wall -Wno-array-bounds -Wunknown-pragmas"            CACHE STRING "C Flags used for debug compilation mode")

# Mechanism via FindLIB.cmake
#set( BOOST_ROOT     "/export/contrib/boost-1.48.0/linux64"    CACHE STRING "Boost search location" )
# In windows, this HAS TO BE set. In Linux, probably not?
set( BOOST_ROOT     "D:/boost/boost_1_47"    CACHE STRING "Boost search location" )

#set( QTDIR          "/export/contrib/qt-4.7.1/linux64"        CACHE STRING "Qt search location" )
#set( QT_SEARCH_PATH "/export/contrib/qt-4.7.1/linux64" CACHE STRING "QT root dir where to find it" )

# This CAN be set if the automatic script fails.
#set( MATLAB_ROOT     "C:/Program Files (x86)/MATLAB/R2012a/extern"    CACHE STRING "Matlab search location" )

# Mechanism via ConfigLIB.cmake in lib dir
#set( OpenCV_DIR   "/export/contrib/OpenCV-2.3.1/linux64/share/OpenCV"   CACHE STRING "OpenCV config dir, where OpenCVConfig.cmake can be found" )
# In windows, this HAS TO BE set. In Linux, maybe not?
set( OpenCV_DIR   "D:/ocvsvnbin"   CACHE STRING "OpenCV config dir, where OpenCVConfig.cmake can be found" )

#set( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE   CACHE BOOL    "use the rpath setting for all our libs" )
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
#SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

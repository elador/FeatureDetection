# Mechanism via FindLIB.cmake
# ==============================
# In windows, this HAS TO BE set. In Linux, if boost is installed system-wide, it's not necessary. Just leave it (should work), or uncomment.
#set( BOOST_ROOT     "D:/boost/boost_1_47"    CACHE STRING "Boost search location" ) # Windows
#set( BOOST_ROOT     "D:/boost/boost_1_51_0"    CACHE STRING "Boost search location" ) # Windows
#set( BOOST_ROOT     "/export/contrib/boost-1.48.0/linux64"    CACHE STRING "Boost search location" ) # Linux

# This CAN be set if the automatic script fails, but usually it works.
#set( MATLAB_ROOT     "C:/Program Files (x86)/MATLAB/R2012a/extern"    CACHE STRING "Matlab search location" )
#Note: I think this path should be given without "/extern", but I can't test it since it works on my system.
set( MATLAB_ROOT     "/home/poschmann/workspaces/RTL/dependencies/matlab2"    CACHE STRING "Matlab search location" ) # Linux

# Mechanism via ConfigLIB.cmake in lib dir
# ==============================
# In windows, this HAS TO BE set. In Linux, it's usually not necessary. Just leave it (should work), or uncomment.
#set( OpenCV_DIR   "D:/ocvsvnbin"   CACHE STRING "OpenCV config dir, where OpenCVConfig.cmake can be found" ) # Windows
#set( OpenCV_DIR   "/export/contrib/OpenCV-2.3.1/linux64/share/OpenCV"   CACHE STRING "OpenCV config dir, where OpenCVConfig.cmake can be found" ) # Linux
set( OpenCV_DIR   "/opt/OpenCV-2.3.1/install/share/OpenCV"   CACHE STRING "OpenCV config dir, where OpenCVConfig.cmake can be found" ) # Linux

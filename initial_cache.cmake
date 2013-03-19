# Mechanism via FindLIB.cmake
# ==============================
# In windows, this HAS TO BE set. In Linux, if boost is installed system-wide, it's not necessary. Just leave it (should work), or uncomment.
set( BOOST_ROOT     "C:\\boost\\install"    CACHE STRING "Boost search location" ) # Windows, self-built
#set( BOOST_ROOT     "/home/patrik/boost/install"    CACHE STRING "Boost search location" ) # Linux, self-built

# This CAN be set if the automatic script fails, but usually it works:
set( MATLAB_ROOT     "C:/Program Files/MATLAB/R2012b/"    CACHE STRING "Matlab search location" ) #Note: Win: This path should be given without "/extern"
#set( MATLAB_ROOT     "/home/poschmann/workspaces/RTL/dependencies/matlab2"    CACHE STRING "Matlab search location" ) # Linux

#set( HDF5_ROOT "C:\\Program Files\\HDF5" CACHE STRING "HDF5 search location" ) # Windows

# Mechanism via ConfigLIB.cmake in lib dir
# ==============================
# In windows, this HAS TO BE set. In Linux, it's usually not necessary. Just leave it (should work), or uncomment.
set( OpenCV_DIR   "C:\\opencv\\install"   CACHE STRING "OpenCV config dir, where OpenCVConfig.cmake can be found" ) # Windows, self-built
#set( OpenCV_DIR   "/home/patrik/opencv/install/share/OpenCV"   CACHE STRING "OpenCV config dir, where OpenCVConfig.cmake can be found" ) # Linux, openCV from git (self-built)

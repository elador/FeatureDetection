echo "This sets up a Debug and Release build for use with linux Makefiles. If you are using another CMake generator (e.g. to build Eclipse project files), don't use this."
echo "======================================================================================="


mkdir Debug && cd Debug
cmake -C ../FeatureDetection/initial_cache.cmake -D CMAKE_BUILD_TYPE=Debug -D ../FeatureDetection/
#if we want more compiler warnings:
#cmake -C ../FeatureDetection/initial_cache.cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Wconversion" ../FeatureDetection/

cd ..
mkdir Release && cd Release
cmake -C ../FeatureDetection/initial_cache.cmake -D CMAKE_BUILD_TYPE=Release -D ../FeatureDetection/

cd ..
echo "
ALL: build

build: Release Debug

Release:
	cd Release && make

Debug:
	cd Debug && make
" > Makefile
echo "Directories for Debug and Release builds are now ready."
echo "Run 'make Debug', 'make Release' or just 'make' to build both."

@ECHO OFF
echo.
echo 1) SETUP BUILD:
echo ============================
echo Running:
echo ---
echo cmake -C ..\FeatureDetection\initial_cache.cmake -G "Visual Studio 12 Win64" ..\FeatureDetection
cmake -C ..\FeatureDetection\initial_cache.cmake -G "Visual Studio 12 Win64" ..\FeatureDetection
echo ---
echo from the build directory (you should currently be in the build directory).
echo.
echo 2) COMPILE:
echo ============================
echo Open the .sln and compile in Visual Studio.
echo.
echo 3) RUN:
echo ============================
echo To run the apps, you should add the following paths (please change it to reflect your paths) to your windows PATH variable:
echo D:\ocvsvnbin\bin\Release;D:\ocvsvnbin\bin\Debug;D:\boost\boost_1_47\lib;C:\Program Files (x86)\MATLAB\R2012a\bin\win32
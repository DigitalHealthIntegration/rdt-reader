@REM generate the VS project
cmake -G "Visual Studio 15 2017 Win64" -B _build
@REM also build it
cmake --build _build
pause
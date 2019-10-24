@REM generate the VS project
mkdir _build
cmake -G "Visual Studio 15 2017 Win64" -B _build -S .
@REM also build it
cmake --build _build
pause
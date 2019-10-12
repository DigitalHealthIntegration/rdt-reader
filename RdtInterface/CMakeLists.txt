# CMake looks for build instructions in a file called CMakeLists.txt. Always
# start the top level CMakeLists.txt with the minimum required CMake version.
# This hints to the build system and future developers which features are
# currently in use.

cmake_minimum_required(VERSION 3.2)

include("cmake/HunterGate.cmake")

HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.216.tar.gz"
    SHA1 "a492a1879bee60c29f7af17ef56b45f0d7766241"
	LOCAL
)

project(RDTFilterInt)



SET(SOURCE
main.cpp
RdtReader.cpp
)

SET(HEADERS
RdtProcessing.h
RdtReader.h
)

# DOCUMENTATION_START {
hunter_add_package(OpenCV)
find_package(OpenCV REQUIRED)

add_executable(RDTFilterInt ${SOURCE} ${HEADERS})
target_link_libraries(RDTFilterInt PRIVATE ${OpenCV_LIBS})
# DOCUMENTATION_END }

message("OpenCV_DIR: ${OpenCV_DIR}")
message("OpenCV_CONFIG: ${OpenCV_CONFIG}")
message("OpenCV_LIBS: ${OpenCV_LIBS}")

cmake_minimum_required(VERSION 3.12)
set(PROGRAM DecisionTree)
project(${PROGRAM})

set(CMAKE_CXX_STANDARD 17) # ROOT needs C++17 still :(
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ROOT REQUIRED)

include(${ROOT_USE_FILE})

# Add executable
add_executable(${PROGRAM} Main.cpp DecisionTree.cpp TreeNode.cpp Dataframe.cpp)

# Link against ROOT
target_link_libraries(${PROGRAM} PRIVATE ${ROOT_LIBRARIES})

# COMPILER OPTIONS
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(INFO " GCC Debug build")
        target_compile_options(${PROGRAM} PRIVATE "-Wall" "-Wpedantic" "-Wextra" "-Wunused-value" "-Wunused-function" "-Wshadow" "-g" "-O0" "-std=c++20" "-Wno-cpp")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(INFO " GCC Release build")
        target_compile_options(${PROGRAM} PRIVATE "-Wall" "-Wpedantic" "-Wextra" "-Wunused-value" "-Wunused-function" "-Wshadow" "-Ofast" "-std=c++20" "-Wno-cpp")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(INFO " Clang Debug build")
        target_compile_options(${PROGRAM} PRIVATE -g -O0 -Wall -Wextra -Wpedantic -std=c++20 -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(${PROGRAM} PRIVATE -fsanitize=address)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(INFO " Clang Release build")
        target_compile_options(${PROGRAM} PRIVATE "-Wall" "-Wextra" "-Wpedantic" "-O3" "-std=c++20")
    endif()
endif()
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})


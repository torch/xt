cmake_minimum_required(VERSION 3.0)
project(XT)

# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

# Nice trace in debug mode
# see https://github.com/bombela/backward-cpp
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_subdirectory(lib/backtrace)
endif()

# TH stuff
add_subdirectory(lib/TH)
include_directories(${CMAKE_SOURCE_DIR}/lib/TH ${CMAKE_BINARY_DIR}/lib/TH)
find_package(CUDA 5.5)
if(CUDA_FOUND)
  add_definitions(-DXT_HAS_CUDA)
  include_directories(${CMAKE_SOURCE_DIR}/lib/THC ${CMAKE_BINARY_DIR}/lib/THC ${CUDA_INCLUDE_DIRS})
  add_subdirectory(lib/THC)
else()
  message(STATUS "CUDA not found: disabling THC")
endif()

add_subdirectory(src/tensor)

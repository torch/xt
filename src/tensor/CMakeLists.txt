# Currently, we assume that long is 64 bits (TH legacy)
include(CheckTypeSize)
check_type_size(long SIZEOF_LONG LANGUAGE CXX)
if(NOT SIZEOF_LONG EQUAL 8)
  message(FATAL_ERROR "Long size is ${SIZEOF_LONG}; 8 expected.")
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
"
#include <iostream>
#include <iomanip>

int main()
{
  std::cout << std::defaultfloat;
  return 0;
}
"
HAS_DEFAULTFLOAT
)

add_subdirectory(lua)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xt)
configure_file(Tensor.h ${CMAKE_CURRENT_BINARY_DIR}/xt COPYONLY)
configure_file(Context.h ${CMAKE_CURRENT_BINARY_DIR}/xt COPYONLY)
configure_file(dispatch.h ${CMAKE_CURRENT_BINARY_DIR}/xt COPYONLY)
configure_file(xttensor.h ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/xt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xt/TensorTH.h ${CMAKE_CURRENT_BINARY_DIR}/TensorTH.cc
  COMMAND lua ARGS tensor.lua ${CMAKE_CURRENT_BINARY_DIR}/xt/TensorTH.h ${CMAKE_CURRENT_BINARY_DIR}/TensorTH.cc ${CUDA_FOUND}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS lua tensor.lua interface.lua record.lua types.lua
)

set(src
  ${CMAKE_CURRENT_BINARY_DIR}/TensorTH.cc
  Context.cc
  Tensor.cc
  TensorOperator.cc
  TensorPrint.cc
  ${CMAKE_CURRENT_BINARY_DIR}/xt/TensorTH.h
)

if(HAS_DEFAULTFLOAT)
  add_definitions(-DHAS_DEFAULTFLOAT)
endif()

add_library(xttensor SHARED ${src})
target_link_libraries(xttensor TH)
if(CUDA_FOUND)
  target_link_libraries(xttensor THC)
endif()
add_executable(test-basic test/basic.cc)
target_link_libraries(test-basic xttensor)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xt DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xttensor.h DESTINATION include)
install(TARGETS xttensor LIBRARY DESTINATION lib)
install(TARGETS test-basic RUNTIME DESTINATION share/xt/tensor)

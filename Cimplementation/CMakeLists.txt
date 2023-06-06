cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(C_modelLoader)

find_package(Torch REQUIRED)

add_executable(C_modelLoader tracedModel_loader.cpp)
target_link_libraries(C_modelLoader "${TORCH_LIBRARIES}")
set_property(TARGET C_modelLoader PROPERTY CXX_STANDARD 14)
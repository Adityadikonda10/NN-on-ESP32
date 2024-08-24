cmake_minimum_required(VERSION 3.5)

project(test_model)

set(CMAKE_CXX_STANDARD 11)

# Add the executable
add_executable(test_model test_model.cpp)

# Manually add the path to the jsoncpp include directory
include_directories(/usr/local/include)

# Link the jsoncpp library
target_link_libraries(test_model /usr/local/lib/libjsoncpp.dylib)

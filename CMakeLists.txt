cmake_minimum_required(VERSION 3.8.0)
 
project(ceres_example)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(
  ${CERES_INCLUDE_DIRS}
)
include_directories(
  ${YAML_CPP_INCLUDE_DIR}
) 
include_directories(
  "/usr/lnclude/eigen3"
) 
include_directories(
  "./src/include"
) 
include_directories(
  ${PYTHON_INCLUDE_DIRS}
) 
add_executable(ceres_example
src/ceres_example.cpp)
 
target_link_libraries(ceres_example
    ${CERES_LIBRARIES}
)
target_link_libraries(ceres_example
    yaml-cpp
)
target_link_libraries(ceres_example
    ${PYTHON_LIBRARIES}
)

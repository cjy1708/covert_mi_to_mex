cmake_minimum_required(VERSION 3.26)
project(covert_to_mex)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCE_C "src/*.c")
# 包含头文件，mi_utils.h
include_directories(INCLUDE ${PROJECT_SOURCE_DIR}/. /home/mi/Software/MATLAB/R2022b/extern/include/.)

add_executable(${PROJECT_NAME}
        ${INCLUDE}
        ${SOURCE_C}
        MiHigherDim.cpp
)

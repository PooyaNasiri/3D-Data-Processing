cmake_minimum_required (VERSION 2.6)

project (sgm LANGUAGES C CXX)
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(.)

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11  --compiler-options -Wall")
#set(CMAKE_CUDA_ARCHITECTURES "native")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    # extend release-profile with fast-math
endif()

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()


# For compilation ...
# Specify target & source files to compile it from
add_executable(
    sgm
    sgm.h
    sgm.cpp
    main.cpp
    )

#set_property(TARGET ACMH PROPERTY CUDA_ARCHITECTURES native)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(sgm
    ${OpenCV_LIBS}
    )

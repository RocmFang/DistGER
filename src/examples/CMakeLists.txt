# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_BUILD_TYPE=Release")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_app_exec(huge_walk)
#add_app_exec(biased_walk)
#add_app_exec(dynamic_walk)

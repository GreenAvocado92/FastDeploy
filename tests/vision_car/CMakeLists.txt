PROJECT(FastDeploy_Test C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(/usr/local/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(FastDeploy_Test main.cpp)
target_link_libraries(FastDeploy_Test ${FASTDEPLOY_LIBS})
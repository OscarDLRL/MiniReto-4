cmake_minimum_required(VERSION 3.8)
project(vision_puzzlebot_trt)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================
# Dependencias ROS 2
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# =========================
# OpenCV
# =========================
find_package(OpenCV REQUIRED)

# =========================
# TensorRT / CUDA paths
# =========================
include_directories(
  include
  /usr/include/aarch64-linux-gnu
  /usr/local/cuda/include
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  /usr/lib/aarch64-linux-gnu
  /usr/local/cuda/lib64
)

# =========================
# Ejecutable
# =========================
add_executable(sign_detect_trt src/sign_detect_trt.cpp)

ament_target_dependencies(sign_detect_trt
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

target_link_libraries(sign_detect_trt
  ${OpenCV_LIBS}
  nvinfer
  nvonnxparser
  cudart
)

# =========================
# Instalar ejecutable
# =========================
install(TARGETS
  sign_detect_trt
  DESTINATION lib/${PROJECT_NAME}
)

# =========================
# Instalar modelos
# =========================
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

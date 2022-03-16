cmake_minimum_required(VERSION 3.12)
project(torchsparse_dynamic)
set(CMAKE_CXX_STANDARD 17)

if(BUILD_TEST)
  enable_testing()
endif()

file(GLOB HEADERS src/*.h)
file(GLOB SOURCES src/*.cpp)
file(GLOB TEST_SOURCES test/*.cpp)

find_package(Python3 COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
target_include_directories(${PROJECT_NAME} PUBLIC src)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

if(BUILD_TEST)
  add_executable(test_dynamic ${TEST_SOURCES})
  target_link_libraries(test_dynamic PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(test_dynamic PRIVATE Python3::Python)
  target_link_libraries(test_dynamic PRIVATE ${PROJECT_NAME})
  target_include_directories(test_dynamic PRIVATE test)
  target_include_directories(test_dynamic PRIVATE src)
endif()

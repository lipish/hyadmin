set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES 89 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --threads=0")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "PYTHON3_EXECUTABLE: ${Python3_EXECUTABLE}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TORCH_LIB_DIR: ${TORCH_LIB_DIR}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "TORCH_INCLUDE_DIR: ${TORCH_INCLUDE_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${TORCH_INCLUDE_DIR})
link_directories(${TORCH_LIB_DIR})

aux_source_directory(cpu_backend CPU_BACKEND_SOURCES)
aux_source_directory(perfetto PERFETTO_SOURCES)
aux_source_directory(. SOURCES)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/operators/moe)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/operators/fp8)

pybind11_add_module(${PROJECT_NAME} MODULE
    ${CPU_BACKEND_SOURCES}
    ${PERFETTO_SOURCES}
    ${SOURCES}
)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../torch/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE moe fp8)
target_link_libraries(${PROJECT_NAME} PRIVATE ${NUMA_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE torch_python)
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NUMA)
target_compile_definitions(${PROJECT_NAME} PRIVATE _GNU_SOURCE)

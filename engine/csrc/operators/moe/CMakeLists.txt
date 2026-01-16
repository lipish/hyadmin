file(GLOB_RECURSE MOE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
)

add_library(moe STATIC ${MOE_SOURCES})
set(MOE_FLAGS
    -flax-vector-conversions -fopenmp
    -mf16c -mfma -mavx -mavx2
    -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vpopcntdq -mavx512bf16
)
target_compile_options(moe PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MOE_FLAGS}>")
target_compile_options(moe PRIVATE "$<$<COMPILE_LANGUAGE:C>:${MOE_FLAGS}>")
target_compile_definitions(moe PRIVATE USE_NUMA)
target_compile_definitions(moe PRIVATE _GNU_SOURCE)
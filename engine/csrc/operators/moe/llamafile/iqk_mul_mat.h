#pragma once
#include <stdbool.h>
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_compute_params;

bool iqk_mul_mat(long, long, long,int, const void*, long, int, const void*, long,float*, long, int, int);

#ifdef __cplusplus
}
#endif
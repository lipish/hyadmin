// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/sgemm.h
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#pragma once
#include <stdbool.h>
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_compute_params;

bool iqk_mul_mat(long, long, long,int, const void*, long, int, const void*, long,float*, long, int, int);

bool llamafile_sgemm(long, long, long, const void*, long, const void*, long, void*, long, int, int, int, int, int, int, int, const void* S=nullptr, const int bias_m=0);

#ifdef __cplusplus
}
#endif

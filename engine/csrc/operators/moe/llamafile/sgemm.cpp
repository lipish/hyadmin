// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/tinyblas_cpu_sgemm_amd_zen4.cpp
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#include "sgemm.h"
#include "iqk_mul_mat.h"
#include "tinyblas_cpu.h"

namespace {

template <typename TC>
bool llamafile_sgemm_impl(long m, long n, long k, const void *A, long lda,
                          const void *B, long ldb, TC *C, long ldc, int ith,
                          int nth, int task, int Atype, int Btype, int Ctype,
                          int precision, const void *S, const int bias_m) {
  switch (Atype) {
  case GGML_TYPE_F32: {
    if (Btype != GGML_TYPE_F32)
      return NOT_SUPPORTED;
#if defined(__AVX512F__)
    if (k % 16)
      return NOT_SUPPORTED;
    tinyBLAS<0, 16, __m512, __m512, float, float, TC> tb{
        k, (const float *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
    tb.matmul(m, n, task);
    return true;
#elif defined(__AVX__) || defined(__AVX2__)
    if (k % 8)
      return NOT_SUPPORTED;
    tinyBLAS<0, 8, __m256, __m256, float, float, TC> tb{
        k, (const float *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
    tb.matmul(m, n, task);
    return true;
#else
    return NOT_SUPPORTED;
#endif
  }
  case GGML_TYPE_F8_E4M3: {
#if defined(__AVX512BF16__)
    if (k % 32)
      return NOT_SUPPORTED;
    if (Btype != GGML_TYPE_BF16) {
      return NOT_SUPPORTED;
    }
#ifndef FLAG_PRECISE
    tinyBLAS<0, 32, __m512, __m512bh, ggml_f8_e4m3_t, ggml_bf16_t, TC> tb{
        k,   (const ggml_f8_e4m3_t *)A,
        lda, (const ggml_bf16_t *)B,
        ldb, C,
        ldc, ith,
        nth, (const float *)S, bias_m};
    tb.matmul(m, n, task);
    return true;
#endif // FLAG_PRECISE
#else
    return NOT_SUPPORTED;
#endif
  }

  case GGML_TYPE_BF16: {
#if defined(__AVX512BF16__)
    if (k % 32)
      return NOT_SUPPORTED;
    if (Btype == GGML_TYPE_F32 && n < 2) {
      tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, float, TC> tb{
          k,  (const ggml_bf16_t *)A, lda, (const float *)B, ldb, C, ldc, ith,
          nth};
      tb.matmul(m, n, task);
      return true;
    }
    if (Btype == GGML_TYPE_F32)
      return WANT_QUANTIZATION;
    if (Btype != GGML_TYPE_BF16)
      return NOT_SUPPORTED;
#ifndef FLAG_PRECISE
    // bf16 goes here
    tinyBLAS<0, 32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, TC> tb{
        k,   (const ggml_bf16_t *)A,
        lda, (const ggml_bf16_t *)B,
        ldb, C,
        ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
#else
    tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, TC> tb{
        k,   (const ggml_bf16_t *)A,
        lda, (const ggml_bf16_t *)B,
        ldb, C,
        ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
#endif
#elif defined(__AVX512F__)
    if (k % 16)
      return NOT_SUPPORTED;
    tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, float, TC> tb{
        k,  (const ggml_bf16_t *)A, lda, (const float *)B, ldb, C, ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
#elif defined(__AVX2__)
    if (k % 8)
      return NOT_SUPPORTED;
    if (Btype != GGML_TYPE_F32)
      return NOT_SUPPORTED;
    tinyBLAS<0, 8, __m256, __m256, ggml_bf16_t, float, TC> tb{
        k,  (const ggml_bf16_t *)A, lda, (const float *)B, ldb, C, ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
#else
    return NOT_SUPPORTED;
#endif
  }

  case GGML_TYPE_F16: {
#if defined(__AVX512F__)
    if (k % 16)
      return NOT_SUPPORTED;
    if (Btype == GGML_TYPE_F32 && n < 2) {
      tinyBLAS<0, 16, __m512, __m512, ggml_fp16_t, float, TC> tb{
          k,  (const ggml_fp16_t *)A, lda, (const float *)B, ldb, C, ldc, ith,
          nth};
      tb.matmul(m, n, task);
      return true;
    }
    if (Btype == GGML_TYPE_F32)
      return WANT_QUANTIZATION;
    if (Btype != GGML_TYPE_F16)
      return NOT_SUPPORTED;
    tinyBLAS<0, 16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, TC> tb{
        k,   (const ggml_fp16_t *)A,
        lda, (const ggml_fp16_t *)B,
        ldb, C,
        ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
    // if (X86_CHECK(F16C)) {
    if (k % 8)
      return NOT_SUPPORTED;
    if (Btype == GGML_TYPE_F32 && n < 2) {
      tinyBLAS<0, 8, __m256, __m256, ggml_fp16_t, float, TC> tb{
          k,  (const ggml_fp16_t *)A, lda, (const float *)B, ldb, C, ldc, ith,
          nth};
      tb.matmul(m, n, task);
      return true;
    }
    if (Btype == GGML_TYPE_F32)
      return WANT_QUANTIZATION;
    if (Btype != GGML_TYPE_F16)
      return NOT_SUPPORTED;
    tinyBLAS<0, 8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, TC> tb{
        k,   (const ggml_fp16_t *)A,
        lda, (const ggml_fp16_t *)B,
        ldb, C,
        ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
    // } else {
    //     return NOT_SUPPORTED;
    // }
#else
    return NOT_SUPPORTED;
#endif
  }

  case GGML_TYPE_Q8_0: {
    if (Btype == GGML_TYPE_F32)
      return WANT_QUANTIZATION;
    if (Btype != GGML_TYPE_Q8_0)
      return NOT_SUPPORTED;
#if defined(__AVX2__) || defined(__AVX512F__)
    tinyBLAS_Q0_AVX2<0, block_q8_0, block_q8_0, TC> tb{
        k,  (const block_q8_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
#else
    return NOT_SUPPORTED;
#endif
  }

  case GGML_TYPE_Q4_0: {
    if (Btype == GGML_TYPE_F32)
      return WANT_QUANTIZATION;
    if (Btype != GGML_TYPE_Q8_0)
      return NOT_SUPPORTED;
#if defined(__AVX2__) || defined(__AVX512F__)
    tinyBLAS_Q0_AVX2<0, block_q4_0, block_q8_0, TC> tb{
        k,  (const block_q4_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith,
        nth};
    tb.matmul(m, n, task);
    return true;
#else
    return NOT_SUPPORTED;
#endif
  }

  default:
    return NOT_SUPPORTED;
  }

  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  (void)ith;
  (void)nth;
  (void)Atype;
  (void)Btype;
  (void)precision;
}

} // namespace

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32,
 *                     GGML_PREC_DEFAULT);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @param precision may be used to control the internal compute type
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(long m, long n, long k, const void *A, long lda,
                     const void *B, long ldb, void *C, long ldc, int ith,
                     int nth, int task, int Atype, int Btype, int Ctype,
                     int precision, const void *S, int bias_m) {
  assert(m >= 0);
  assert(n >= 0);
  assert(k >= 0);
  assert(lda >= k);
  assert(ldb >= k);
  assert(ldc >= m);
  assert(nth > 0);
  assert(ith < nth);

#if QK_K == 256
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) &&                                                       \
    (defined(__FMA__) ||                                                       \
     (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))))
  /*
  moonll
  more Btype accept
  }*/

  if (Ctype == GGML_TYPE_F32) {
    if (iqk_mul_mat(m, n, k * ggml_blck_size(ggml_type(Atype)), Atype, A, lda,
                    Btype, B, ldb, (float *)C, ldc, ith, nth)) {
      return true;
    }
  }

#endif
#endif
#endif

  switch (Ctype) {
  case GGML_TYPE_F32:
    return llamafile_sgemm_impl(m, n, k, A, lda, B, ldb, (float *)C, ldc, ith,
                                nth, task, Atype, Btype, Ctype, precision, S, bias_m);
  default:
    return NOT_SUPPORTED;
  }
}

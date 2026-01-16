#include "ops.h"

#include <cuda_fp8.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

template<typename T>
__global__ void fp8_gemv_kernel(
    const at::Float8_e4m3fn* __restrict__ A,
    const float* __restrict__ a_s,
    const at::Float8_e4m3fn* __restrict__ B,
    const float* __restrict__ b_s,
    T* __restrict__ C,
    const int K) {
    const uint32_t* __restrict__ a = (uint32_t*)A;
    const uint32_t* __restrict__ b = (uint32_t*)B;
    a += threadIdx.x;
    b += blockIdx.x * K / 4 + threadIdx.x;
    b_s += blockIdx.x / 128 * (K / 128);
    float cs = 0;
    for (int i = 0; i < K / 128; i++) {
#if __CUDA_ARCH__ >= 1000
        auto a_p = __ldg(a + i * 32);
        auto a_p_0 = __nv_cvt_fp8x2_to_halfraw2(*(__nv_fp8x2_storage_t*)&a_p, __NV_E4M3);
        auto a_p_1 = __nv_cvt_fp8x2_to_halfraw2(*((__nv_fp8x2_storage_t*)&a_p+1), __NV_E4M3);
        auto b_p = __ldcs(b + i * 32);
        auto b_p_0 = __nv_cvt_fp8x2_to_halfraw2(*(__nv_fp8x2_storage_t*)&b_p, __NV_E4M3);
        auto b_p_1 = __nv_cvt_fp8x2_to_halfraw2(*((__nv_fp8x2_storage_t*)&b_p+1), __NV_E4M3);
        float cs_p;
        asm("fma.rn.f32.f16 %0, %1, %2, .0;" : "=f"(cs_p) : "h"(a_p_0.x), "h"(b_p_0.x));
        asm("fma.rn.f32.f16 %0, %1, %2, %3;" : "=f"(cs_p) : "h"(a_p_0.y), "h"(b_p_0.y), "f"(cs_p));
        asm("fma.rn.f32.f16 %0, %1, %2, %3;" : "=f"(cs_p) : "h"(a_p_1.x), "h"(b_p_1.x), "f"(cs_p));
        asm("fma.rn.f32.f16 %0, %1, %2, %3;" : "=f"(cs_p) : "h"(a_p_1.y), "h"(b_p_1.y), "f"(cs_p));
        cs += cs_p * a_s[i] * b_s[i];
#else
        auto a_p_r = __ldg(a + i * 32);
        auto b_p_r = __ldcs(b + i * 32);
        auto a_p = float4(*(__nv_fp8x4_e4m3*)&a_p_r);
        auto b_p = float4(*(__nv_fp8x4_e4m3*)&b_p_r);
        cs += (a_p.x * b_p.x + a_p.y * b_p.y + a_p.z * b_p.z + a_p.w * b_p.w) * a_s[i] * b_s[i];
#endif
    }
    for (int offset = 1; offset <= 16; offset *= 2) {
        cs += __shfl_down_sync(0xFFFFFFFF, cs, offset);
    }
    if (threadIdx.x == 0) {
        if constexpr (std::is_same_v<T, at::BFloat16>) {
            nv_bfloat16* __restrict__ c = (nv_bfloat16*)C;
            c[blockIdx.x] = __float2bfloat16_rz(cs);
        }
        else {
            C[blockIdx.x] = cs;
        }
    }
}

void fp8_gemv(torch::Tensor& a, torch::Tensor& a_s, torch::Tensor& b, torch::Tensor& b_s, torch::Tensor& c, int N, int K) {
    at::cuda::CUDAGuard device_guard{c.device()};
    auto s = at::cuda::getCurrentCUDAStream().stream();
    if (c.dtype() == at::kBFloat16) {
        fp8_gemv_kernel<<<N, 32, 0, s>>>(a.data_ptr<at::Float8_e4m3fn>(), a_s.data_ptr<float>(), b.data_ptr<at::Float8_e4m3fn>(), b_s.data_ptr<float>(), c.data_ptr<at::BFloat16>(), K);
    }
    else {
        fp8_gemv_kernel<<<N, 32, 0, s>>>(a.data_ptr<at::Float8_e4m3fn>(), a_s.data_ptr<float>(), b.data_ptr<at::Float8_e4m3fn>(), b_s.data_ptr<float>(), c.data_ptr<float>(), K);
    }
}
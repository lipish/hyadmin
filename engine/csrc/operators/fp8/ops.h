#pragma once

#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

void fp8_gemv(torch::Tensor& a, torch::Tensor& a_s, torch::Tensor& b, torch::Tensor& b_s, torch::Tensor& c, int N, int K);

/**
 * @Description  :
 * @Author       : chenht2022, Jianwei Dong
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : Jianwei Dong
 * @LastEditTime : 2024-08-26 22:47:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include "cpu_backend/cpuinfer.h"
#include "operators/moe/moe.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <torch/library.h>
#include <torch/torch.h>
#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include "operators/fp8/ops.h"

namespace py = pybind11;
using namespace pybind11::literals;

class MOEBindings {
  public:
    class WarmUpBindinds {
      public:
        struct Args {
            /* 
             * CPUInfer* must be the first field in Args struct.
             * This is required because submit() and cuda_launch_host_func()
             * in cpuinfer.h will treat the Args* pointer as CPUInfer**
             * and only modify the first sizeof(CPUInfer*) bytes.
             */
            heyi::CPUInfer *cpuinfer; 
            int task_id;
            heyi::MOE *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(args_->task_id, &heyi::MOE::warm_up, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> wrapped_warmup(heyi::MOE &moe, int task_id) {
            Args *args = new Args{nullptr, task_id, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            heyi::CPUInfer *cpuinfer;
            int task_id;
            heyi::MOE *moe;
            int qlen;
            int k;
            const uint64_t *expert_ids;
            const float *weights;
            const void *input;
            void *output;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                args_->task_id,
                &heyi::MOE::forward, args_->moe, args_->qlen, args_->k,
                args_->expert_ids, args_->weights, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t>
        wrapped_forward(heyi::MOE &moe, int task_id, int qlen, int k, intptr_t expert_ids,
                        intptr_t weights, intptr_t input, intptr_t output) {
            Args *args = new Args{nullptr,
                                  task_id,
                                  &moe,
                                  qlen,
                                  k,
                                  (const uint64_t *)expert_ids,
                                  (const float *)weights,
                                  (const void *)input,
                                  (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class GetWeightBindings {
      public:
        struct Args {
            heyi::CPUInfer *cpuinfer;
            int task_id;
            heyi::MOE *moe;
            int iexpert;
            intptr_t gate_proj;
            intptr_t up_proj;
            intptr_t down_proj;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                args_->task_id, 
                &heyi::MOE::get_weight, 
                args_->moe,
                args_->iexpert,
                args_->gate_proj, 
                args_->up_proj,
                args_->down_proj
            );
        }
        static std::pair<intptr_t, intptr_t> wrapped_getweight(
            heyi::MOE &moe, int task_id, int iexpert, 
            intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj
        ) {
            Args *args = new Args{
                nullptr,
                task_id,
                &moe,
                iexpert,
                gate_proj,
                up_proj,
                down_proj,
            };
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class SyncBindings {
      public:
        struct Args {
            heyi::CPUInfer *cpuinfer;
            int task_id;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->task_queue_->sync(args_->task_id);
        }
        static std::pair<intptr_t, intptr_t> wrapped_sync(heyi::MOE &moe, int task_id) {
            Args *args = new Args{nullptr, task_id};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

void initialize_perfetto() {
    perfetto::TracingInitArgs args;
    args.backends = perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
}


PYBIND11_MODULE(_ext, m) {
    initialize_perfetto();
    
    py::class_<heyi::CPUInfer>(m, "CPUInfer")
        .def(py::init<int, int>())
        .def("start_trace", &heyi::CPUInfer::start_trace)
        .def("end_trace", &heyi::CPUInfer::end_trace)
        .def("submit", &heyi::CPUInfer::submit)
        .def("cuda_launch_host_func", &heyi::CPUInfer::cuda_launch_host_func)
        .def("sync", &heyi::CPUInfer::sync)
        // .def("sync_with_cuda_stream", &heyi::CPUInfer::sync_with_cuda_stream)
        .def("lock", &heyi::CPUInfer::lock)
        .def("unlock", &heyi::CPUInfer::unlock);

    auto moe_module = m.def_submodule("moe");
    py::class_<heyi::MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, 
                        int hidden_size, int intermediate_size,
                        int group_min_len, int group_max_len, 
                        intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj,
                        int gate_type, int up_type, int down_type, int hidden_type,
                        intptr_t gate_inv = 0, intptr_t up_inv = 0, intptr_t down_inv = 0) {  // Default *_inv pointers
            return heyi::MOEConfig(expert_num, routed_expert_num, 
                            hidden_size, intermediate_size,
                            group_min_len, group_max_len, 
                            (void*)gate_proj, (void*)up_proj, (void*)down_proj, 
                            (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type,
                            (void*)gate_inv, (void*)up_inv, (void*)down_inv);
        }), 
        py::arg("expert_num"), py::arg("routed_expert_num"), 
        py::arg("hidden_size"), py::arg("intermediate_size"),
        py::arg("group_min_len"), py::arg("group_max_len"), 
        py::arg("gate_proj"), py::arg("up_proj"), py::arg("down_proj"), 
        py::arg("gate_type"), py::arg("up_type"), py::arg("down_type"), py::arg("hidden_type"),
        py::arg("gate_inv") = 0, py::arg("up_inv") = 0, py::arg("down_inv") = 0  // Only these have default values
    );
    py::class_<heyi::MOE>(moe_module, "MOE")
        .def(py::init<heyi::MOEConfig>())
        .def("wrapped_warmup", &MOEBindings::WarmUpBindinds::wrapped_warmup)
        .def("wrapped_forward", &MOEBindings::ForwardBindings::wrapped_forward)
        .def("wrapped_getweight", &MOEBindings::GetWeightBindings::wrapped_getweight)
        .def("wrapped_sync", &MOEBindings::SyncBindings::wrapped_sync)
        .def("get_weight", &heyi::MOE::get_weight);

    auto fp8_module = m.def_submodule("fp8");
    fp8_module.def("fp8_gemv", &fp8_gemv, "Function to perform fp8 GEMV.",
        py::arg("a"), py::arg("a_s"), py::arg("b"), py::arg("b_s"),
        py::arg("c"), py::arg("N"), py::arg("K"));
}

/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "moe.h"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "perfetto/categories.h"
#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>

using namespace heyi;

void numadist(
    size_t numa_nodes, 
    void* src, std::vector<void*>* dst,
    size_t strided_dim_size, size_t non_strided_dim_size,
    ggml_type type, size_t expert_num
) {
    // printf("Starting numadist function\n");
    
    size_t half_stride = strided_dim_size / 2;
    size_t expert_half_size = half_stride * non_strided_dim_size;
    size_t total_half_size = expert_half_size * expert_num;

    // printf("Computed values:\n");
    // printf("  half_stride: %zu\n", half_stride);
    // printf("  expert_half_size: %zu\n", expert_half_size);
    // printf("  total_half_size: %zu\n", total_half_size);
    
    // Allocate memory for each NUMA node
    for (int i = 0; i < numa_nodes; i++) {
        (*dst)[i] = numa_alloc_onnode(total_half_size * ggml_type_size(type) / ggml_blck_size(type), i);
        // printf("Allocating memory on NUMA node %d: %zu bytes\n", 
        //        i, total_half_size * ggml_type_size(type) / ggml_blck_size(type));
        
        if (!(*dst)[i]) {
            std::cout << "Memory allocation failed on node " << i << std::endl;
        }
    }

    // Perform data distribution
    // printf("Beginning data distribution loop...\n");
    #pragma omp parallel for collapse(2)
    for (size_t inuma = 0; inuma < numa_nodes; inuma++) {
        for (size_t iexpert = 0; iexpert < expert_num; iexpert++) {
            uint8_t* src_ptr = (uint8_t*)src + ((ptrdiff_t)iexpert * strided_dim_size + inuma * half_stride) 
                               * non_strided_dim_size * ggml_type_size(type) / ggml_blck_size(type);
            uint8_t* dst_ptr = (uint8_t*)((*dst)[inuma]) + ((ptrdiff_t)iexpert * half_stride) 
                               * non_strided_dim_size * ggml_type_size(type) / ggml_blck_size(type);

            // printf("  Copying data for:\n");
            // printf("    NUMA node: %d\n", inuma);
            // printf("    Expert: %d\n", iexpert);
            // printf("    Source pointer: %p\n", (void*)src_ptr);
            // printf("    Destination pointer: %p\n", (void*)dst_ptr);
            // printf("    Bytes copied: %zu\n", expert_half_size * ggml_type_size(type) / ggml_blck_size(type));

            memcpy(dst_ptr, src_ptr, expert_half_size * ggml_type_size(type) / ggml_blck_size(type));
        }
    }

    // printf("numadist function completed\n");
}

inline void* numaget(
    size_t expert_id, 
    size_t bias_stride, 
    std::vector<void*> base, 
    size_t strided_dim_size, size_t non_strided_dim_size, 
    ggml_type type
) {
    int numa_node;
    size_t half_stride = strided_dim_size / 2;
    if (bias_stride < half_stride) {
        numa_node = 0;
    } else {
        numa_node = 1;
    }
    size_t bias_stride_ = bias_stride % half_stride;
    size_t offset = (expert_id * half_stride + bias_stride_) * non_strided_dim_size * ggml_type_size(type) / ggml_blck_size(type);
    auto ret = (uint8_t*)base[numa_node] + offset;
    return ret;
}
#endif

inline std::pair<int, int> get_slice(int size, int nth, int ith) {
    // int min_stride = size / nth;
    // int local_stride = min_stride + (ith < (size % nth));
    // int bias_stride = ith * min_stride + std::min(ith, size % nth);
    int local_stride = (ith + 1) * size / nth - ith * size / nth;
    int bias_stride = ith * size / nth;
    return {local_stride, bias_stride};
}

MOE::MOE(MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    gate_inv_ = (float*)config_.gate_inv;
    up_inv_ = (float*)config_.up_inv;
    down_inv_ = (float*)config_.down_inv;

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    gate_proj_numa_.resize(numa_nodes);
    up_proj_numa_.resize(numa_nodes);
    down_proj_numa_.resize(numa_nodes);

    numadist(numa_nodes, config.gate_proj, &gate_proj_numa_, config.intermediate_size, config.hidden_size, config.gate_type, config.expert_num);
    numadist(numa_nodes, config.up_proj, &up_proj_numa_, config.intermediate_size, config.hidden_size, config.up_type, config.expert_num);
    numadist(numa_nodes, config.down_proj, &down_proj_numa_, config.hidden_size, config.intermediate_size, config.down_type, config.expert_num);
    #endif

    std::vector<std::pair<void**, uint64_t>> s_mem_requests;
    s_mem_requests.push_back({(void**)&s_input_fp32_, sizeof(float) * config_.hidden_size});
    s_mem_requests.push_back({(void**)&s_gate_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    s_mem_requests.push_back({(void**)&s_up_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_gate_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_up_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_down_input_[i], config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
        s_mem_requests.push_back({(void**)&s_down_output_[i], sizeof(float) * config_.hidden_size});
    }
    s_mem_requests.push_back({(void**)&s_output_fp32_, sizeof(float) * config_.hidden_size});
    shared_mem_buffer.alloc(this, s_mem_requests);

    std::vector<std::pair<void**, uint64_t>> m_mem_requests;
    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_input_fp32_[i], sizeof(float) * config_.hidden_size});
        m_mem_requests.push_back({(void**)&m_gate_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
        m_mem_requests.push_back({(void**)&m_up_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    }
    m_mem_requests.push_back({(void**)&m_local_gate_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_up_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_intermediate_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_down_input_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_output_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);
}

MOE::~MOE() {
    shared_mem_buffer.dealloc(this);

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    for (int i = 0; i < numa_nodes; i++) {
        numa_free(gate_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type));
        numa_free(up_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type));
        numa_free(down_proj_numa_[i], config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type));
    }
    #endif
}

void MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data(), backend);
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;

    int input_conv_stride = QK_K * std::ceil(1.0f * config_.hidden_size / backend->get_thread_num() / QK_K);
    int input_conv_nth = std::ceil(1.0f * config_.hidden_size / input_conv_stride);

#ifdef USE_NUMA
    if (backend->get_thread_num() % (2 * k) != 0) {
        std::cerr << "invalid thread num " << backend->get_thread_num() << std::endl;
        exit(1);
    };
#endif
    int nth = std::max(1, backend->get_thread_num() / k); // 48cores = 6 * 8experts
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && 
            config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = input;
        } else {
            // TRACE_EVENT_BEGIN("compute", "conv in");
            if (task_id < input_conv_nth) {            
                int ith = task_id;
                int bias = ith * input_conv_stride;
                backend->input_conv_syn[ith].store(0);
        
                uint8_t * input_ptr = (uint8_t *)input + bias * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                float * s_input_fp32_ptr = s_input_fp32_ + bias;
                ggml_type s_gate_input_type = ggml_internal_get_type_traits(config_.gate_type).vec_dot_type;
                uint8_t * s_gate_input_ptr = s_gate_input_ + bias * ggml_type_size(s_gate_input_type) / ggml_blck_size(s_gate_input_type);
                
                to_float(input_ptr, s_input_fp32_ptr, input_conv_stride, config_.hidden_type);
                from_float(s_input_fp32_ptr, s_gate_input_ptr, input_conv_stride, s_gate_input_type);
                backend->input_conv_syn[ith].store(1);
            }
            // TRACE_EVENT_END("compute");

            // TRACE_EVENT_BEGIN("compute", "conv in sync");
            for (int i_chk = 0; i_chk < input_conv_nth; i_chk += 1) {
                while (!backend->input_conv_syn[i_chk].load())
                    __builtin_ia32_pause();
            }
            gate_input_ptr = up_input_ptr = s_gate_input_;
            // TRACE_EVENT_END("compute");
        }

        // gate_input_ptr = up_input_ptr = s_gate_input_;

        // TRACE_EVENT_BEGIN("compute", "up & gate");
        int expert_idx = task_id % k;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id / k;

        backend->interm_conv_grp_syn[expert_idx][ith].store(0);

        auto [local_stride, bias_stride] = get_slice(config_.intermediate_size, nth, ith);
        
        #ifdef USE_NUMA
        void* gate_proj_ptr = numaget(expert_id, bias_stride, gate_proj_numa_, config_.intermediate_size, config_.hidden_size, config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + bias_stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        void* gate_inv_ptr = nullptr;
        if (config_.gate_type == GGML_TYPE_F8_E4M3) {
            gate_inv_ptr = (float_t*)gate_inv_ + (expert_id * config_.intermediate_size / 128 + bias_stride / 128) * config_.hidden_size / 128;
        }

        float* gate_output_ptr = s_gate_output_[expert_idx] + bias_stride;
        // TRACE_EVENT_BEGIN("compute", "gate proj");
        llamafile_sgemm(
            local_stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), 
            gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), 
            gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), 
            gate_output_ptr, local_stride, 
            0, 1, GGML_TASK_TYPE_COMPUTE, 
            config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, 
            GGML_PREC_DEFAULT, 
            gate_inv_ptr, bias_stride
        );
        // TRACE_EVENT_END("compute");
        
        #ifdef USE_NUMA
        void* up_proj_ptr = numaget(expert_id, bias_stride, up_proj_numa_, config_.intermediate_size, config_.hidden_size, config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + bias_stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        void* up_inv_ptr = nullptr;
        if (config_.up_type == GGML_TYPE_F8_E4M3) {
            up_inv_ptr = (float_t*)up_inv_ + (expert_id * config_.intermediate_size / 128 + bias_stride / 128) * config_.hidden_size / 128;
        }

        float* up_output_ptr = s_up_output_[expert_idx] + bias_stride;
        // TRACE_EVENT_BEGIN("compute", "up proj");
        llamafile_sgemm(
            local_stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), 
            up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), 
            up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), 
            up_output_ptr, local_stride, 
            0, 1, GGML_TASK_TYPE_COMPUTE, 
            config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, 
            GGML_PREC_DEFAULT, 
            up_inv_ptr, bias_stride
        );
        // TRACE_EVENT_END("compute");

        // TRACE_EVENT_BEGIN("compute", "act & mult");
        for (int i = bias_stride; i < bias_stride + local_stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        // TRACE_EVENT_END("compute");

        backend->interm_conv_grp_syn[expert_idx][ith].store(1);
        // TRACE_EVENT_BEGIN("compute", "grp sync");
        for (int i_chk = 0; i_chk < nth; i_chk += 1) {
            while (!backend->interm_conv_grp_syn[expert_idx][i_chk].load())
                __builtin_ia32_pause();
        }
        // TRACE_EVENT_END("compute");

        // TRACE_EVENT_BEGIN("compute", "fused conv");
        from_float(s_intermediate_fp32_[expert_idx], s_down_input_[expert_idx], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        // TRACE_EVENT_END("compute");
        // TRACE_EVENT_END("compute");
    }, nullptr);
    nth = backend->get_thread_num();
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        // TRACE_EVENT("compute", "down");
        int ith = task_id;

        auto [local_stride, bias_stride] = get_slice(config_.hidden_size, nth, ith);

        for (int i = bias_stride; i < bias_stride + local_stride; i++) {
            s_output_fp32_[i] = 0;
        }
        // TRACE_EVENT_BEGIN("compute", "down proj");
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            #ifdef USE_NUMA
            void* down_proj_ptr = numaget(expert_id, bias_stride, down_proj_numa_, config_.hidden_size, config_.intermediate_size, config_.down_type);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + bias_stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif

            void* down_inv_ptr = nullptr;
            if (config_.down_type == GGML_TYPE_F8_E4M3) {
                down_inv_ptr = (float_t*)down_inv_ + (expert_id * config_.hidden_size / 128 + bias_stride / 128) * config_.intermediate_size / 128;
            }    
            
            float* down_output_ptr = s_down_output_[expert_idx] + bias_stride;
            llamafile_sgemm(
                local_stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), 
                down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), 
                s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), 
                down_output_ptr, local_stride, 
                0, 1, GGML_TASK_TYPE_COMPUTE, 
                config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, 
                GGML_PREC_DEFAULT, 
                down_inv_ptr, bias_stride
            );
            for (int i = bias_stride; i < bias_stride + local_stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        // TRACE_EVENT_END("compute");
    }, nullptr);
    // TRACE_EVENT_BEGIN("compute", "conv out");
    from_float(s_output_fp32_, output, config_.hidden_size, config_.hidden_type);
    // TRACE_EVENT_END("compute");
}

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);

    int nth = backend->get_thread_num();
    // strided_dim: intermediate_size
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        auto [local_stride, bias_stride] = get_slice(config_.intermediate_size, nth, ith);

        for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx += 1) {
            void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

            #ifdef USE_NUMA
            void* gate_proj_ptr = numaget(expert_idx, bias_stride, gate_proj_numa_, config_.intermediate_size, config_.hidden_size, config_.gate_type);
            #else
            void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + bias_stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
            #endif

            void* gate_inv_ptr = nullptr;
            if (config_.gate_type == GGML_TYPE_F8_E4M3) {
                gate_inv_ptr = (float_t*)gate_inv_ + (expert_idx * config_.intermediate_size / 128 + bias_stride / 128) * config_.hidden_size / 128;
            }

            float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + bias_stride;
            llamafile_sgemm(
                local_stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), 
                gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), 
                gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), 
                gate_output_ptr, config_.intermediate_size, 
                0, 1, GGML_TASK_TYPE_COMPUTE, 
                config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, 
                GGML_PREC_DEFAULT, 
                gate_inv_ptr, bias_stride
            );
            void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

            #ifdef USE_NUMA
            void* up_proj_ptr = numaget(expert_idx, bias_stride, up_proj_numa_, config_.intermediate_size, config_.hidden_size, config_.up_type);
            #else
            void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + bias_stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
            #endif

            void* up_inv_ptr = nullptr;
            if (config_.up_type == GGML_TYPE_F8_E4M3) {
                up_inv_ptr = (float_t*)up_inv_ + (expert_idx * config_.intermediate_size / 128 + bias_stride / 128) * config_.hidden_size / 128;
            }   

            float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + bias_stride;
            llamafile_sgemm(
                local_stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), 
                up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), 
                up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), 
                up_output_ptr, config_.intermediate_size, 
                0, 1, GGML_TASK_TYPE_COMPUTE, 
                config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, 
                GGML_PREC_DEFAULT, 
                up_inv_ptr, bias_stride
            );
            for (int i = 0; i < m_local_num_[expert_idx]; i++) {
                for (int j = bias_stride; j < bias_stride + local_stride; j++) {
                    m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
                }
            }
        }
    }, nullptr);

    // strided_dim: experts_num
    backend->do_work_stealing_job(nth, nullptr, [&](int ith) {
        auto [local_idx, bias_idx] = get_slice(config_.expert_num, nth, ith);
        for (int expert_idx = bias_idx; expert_idx < local_idx + bias_idx; expert_idx += 1) {
            for (int i = 0; i < m_local_num_[expert_idx]; i++) {
                float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size;
                void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
                from_float(intermediate_fp32_ptr, down_input_ptr, config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            }
        }
    }, nullptr);

    // strided_dim: hidden_size
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        auto [local_stride, bias_stride] = get_slice(config_.hidden_size, nth, ith);

        for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx += 1) {
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
            
            #ifdef USE_NUMA
            void* down_proj_ptr = numaget(expert_idx, bias_stride, down_proj_numa_, config_.hidden_size, config_.intermediate_size, config_.down_type);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + bias_stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif

            void* down_inv_ptr = nullptr;
            if (config_.down_type == GGML_TYPE_F8_E4M3) {
                down_inv_ptr = (float_t*)down_inv_ + (expert_idx * config_.hidden_size / 128 + bias_stride / 128) * config_.intermediate_size / 128;
            }    

            float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + bias_stride;
            llamafile_sgemm(
                local_stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), 
                down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), 
                down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), 
                down_output_ptr, config_.hidden_size, 
                0, 1, GGML_TASK_TYPE_COMPUTE, 
                config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, 
                GGML_PREC_DEFAULT, 
                down_inv_ptr, bias_stride
            );
        }
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
}


void numamerge(
    size_t iexpert, 
    std::vector<void*> src, void* dst,
    size_t strided_dim_size, size_t non_strided_dim_size, 
    ggml_type type, size_t numa_nodes, Backend* backend
) {
    size_t half_stride = strided_dim_size / 2;
    size_t expert_half_size = half_stride * non_strided_dim_size;

    const int nth = backend->get_thread_num();
    const int nth_on_node = nth / numa_nodes;

    backend->do_work_stealing_job(nth, nullptr, [&](int ith) {
        size_t inuma = ith * numa_nodes / nth;
        int ith_on_node = ith % nth_on_node;
        auto [local_size, bias_size] = get_slice(expert_half_size, nth_on_node, ith_on_node);
        // printf(
        //     "nth=%d, ith=%d, nth_on_node=%d, ith_on_node=%d, inuma=%lu, "
        //     "cp size=%lu, from src+%lu+%lu to dst+%lu+%lu\n",
        //     nth, ith, nth_on_node, ith_on_node, inuma,
        //     local_size * ggml_type_size(type) / ggml_blck_size(type),
        //     ((ptrdiff_t)iexpert * half_stride) * non_strided_dim_size *
        //         ggml_type_size(type) / ggml_blck_size(type),
        //     bias_size,
        //     ((ptrdiff_t)inuma * half_stride) * non_strided_dim_size *
        //         ggml_type_size(type) / ggml_blck_size(type),
        //     bias_size
        // );

        uint8_t* src_ptr = (uint8_t*)(src[inuma]) + ((ptrdiff_t)iexpert * half_stride) 
                            * non_strided_dim_size * ggml_type_size(type) / ggml_blck_size(type);
        uint8_t* dst_ptr = (uint8_t*)dst + ((ptrdiff_t)inuma * half_stride) 
                            * non_strided_dim_size * ggml_type_size(type) / ggml_blck_size(type);

        memcpy(
            dst_ptr + bias_size * ggml_type_size(type) / ggml_blck_size(type),
            src_ptr + bias_size * ggml_type_size(type) / ggml_blck_size(type),
            local_size * ggml_type_size(type) / ggml_blck_size(type)
        );
    }, nullptr);
}


void MOE::get_weight(int iexpert, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, Backend* backend) {
    // auto start = std::chrono::high_resolution_clock::now();
    int numa_nodes = numa_num_configured_nodes();
    numamerge(iexpert, gate_proj_numa_, (void*)gate_proj, config_.intermediate_size, config_.hidden_size, config_.gate_type, numa_nodes, backend);
    numamerge(iexpert, up_proj_numa_, (void*)up_proj, config_.intermediate_size, config_.hidden_size, config_.up_type, numa_nodes, backend);
    numamerge(iexpert, down_proj_numa_, (void*)down_proj, config_.hidden_size, config_.intermediate_size, config_.down_type, numa_nodes, backend);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;
    // printf("took %f seconds\n", duration.count());
}
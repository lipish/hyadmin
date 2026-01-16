/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "backend.h"

using namespace heyi;

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
thread_local int Backend::numa_node = -1;
#endif

thread_local int Backend::thread_local_id = -1;

Backend::Backend(int max_thread_num) {

    {
        tracing_config_.add_buffers()->set_size_kb(1024 * 256);
        auto* ds_cfg = tracing_config_.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        perfetto::protos::gen::TrackEventConfig track_event_cfg;
        track_event_cfg.add_disabled_categories("*");
        track_event_cfg.add_enabled_categories("compute");
        track_event_cfg.add_enabled_categories("schedule");
        track_event_cfg.add_enabled_categories("taskqueue");
        ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());
    }

    one_shot_mode = false;

    max_thread_num_ = max_thread_num;
    thread_state_.resize(max_thread_num_);
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].curr = std::make_unique<std::atomic<int>>();
        thread_state_[i].status =
            std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::WAITING);
    }
    workers_.resize(max_thread_num_);
    
    bool numa_on = false;
    #ifdef USE_NUMA
    printf("USE_NUMA\n");
    numa_on = true;
    #endif
    
    for (int i = 0; i < max_thread_num_; i++) {
        workers_[i] = std::thread(&Backend::worker_thread, this, i);
        int icpu;
        if (numa_on) {
            icpu = ((double)i / max_thread_num_) * 64;
        } else {
            icpu = i;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(icpu, &cpuset);
        int rc = pthread_setaffinity_np(workers_[i].native_handle(),
                                        sizeof(cpu_set_t), &cpuset);
        printf("bind thread %d to cpu %d, rc=%d\n", i, icpu, rc);
    }

    input_conv_syn = std::vector<std::atomic<int>>(64);

    // up to 8 experts * 64 slices (on up to 512 threads)
    interm_conv_grp_syn = std::vector<std::vector<std::atomic<int>>>(8);
    for (int i = 0; i < 8; i++) {
        interm_conv_grp_syn[i] = std::vector<std::atomic<int>>(64);
    }
}

Backend::~Backend() {
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].status->store(ThreadStatus::EXIT,
                                       std::memory_order_release);
    }
    for (int i = 0; i < max_thread_num_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }
}

void Backend::start_trace(std::string file) {
    tracing_file_ = fopen(file.c_str(), "w");
    tracing_session_ = perfetto::Tracing::NewTrace();
    tracing_session_->Setup(tracing_config_, fileno(tracing_file_));
    tracing_session_->StartBlocking();
}

void Backend::end_trace() {
    // Make sure the last event is closed.
    perfetto::TrackEvent::Flush();
    // Stop tracing and read the trace data.
    tracing_session_->StopBlocking();
    fclose(tracing_file_);
}

int Backend::get_thread_num() { return max_thread_num_; }

void Backend::do_work_stealing_job(int task_num,
                                   std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
#ifdef USE_NUMA
    // numa node location will be calculated based on the number of threads
    thread_num_ = max_thread_num_;
#else
    thread_num_ = std::min(max_thread_num_, task_num);
#endif
    
    one_shot_mode = (task_num <= max_thread_num_);
    
    if (one_shot_mode) {
        for (int i = 0; i < std::min(thread_num_, task_num); i++) {
            thread_state_[i].status->store(ThreadStatus::WORKING,
                                           std::memory_order_release);
        }
    } else {
        int base = task_num / thread_num_;
        int remain = task_num % thread_num_;
        thread_state_[0].end = base + (0 < remain);
        thread_state_[0].curr->store(0, std::memory_order_relaxed);
        thread_state_[0].status->store(ThreadStatus::WORKING,
                                    std::memory_order_release);
        for (int i = 1; i < thread_num_; i++) {
            thread_state_[i].curr->store(thread_state_[i - 1].end,
                                        std::memory_order_relaxed);
            thread_state_[i].end = thread_state_[i - 1].end + base + (i < remain);
            thread_state_[i].status->store(ThreadStatus::WORKING,
                                        std::memory_order_release);
        }
    }
    
    for (int i = 0; i < thread_num_; i++) {
        uint64_t sleepy = 0;
        while (thread_state_[i].status->load(std::memory_order_acquire) ==
               ThreadStatus::WORKING) 
            sleepy += 1;
            if (sleepy >= 4400000000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
    }
}

void Backend::process_tasks(int thread_id) {
    
    #ifdef USE_NUMA
    if(numa_node == -1){
        numa_node = thread_id * numa_num_configured_nodes() / thread_num_;
        struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
        numa_bitmask_setbit(mask, numa_node);
        numa_bind(mask);
    }
    #endif

    if (init_func_ != nullptr) {
        init_func_(thread_id);
    }
    
    if (one_shot_mode) {
        TRACE_EVENT_BEGIN("schedule", "own");
        compute_func_(thread_id);
        TRACE_EVENT_END("schedule");
    } else {
        while (true) {
            int task_id = thread_state_[thread_id].curr->fetch_add(
                1, std::memory_order_acq_rel);
            if (task_id >= thread_state_[thread_id].end) {
                break;
            }
            TRACE_EVENT_BEGIN("schedule", "own");
            compute_func_(task_id);
            TRACE_EVENT_END("schedule");
        }
    }
    
    if (finalize_func_ != nullptr) {
        finalize_func_(thread_id);
    }
    thread_state_[thread_id].status->store(ThreadStatus::WAITING,
                                           std::memory_order_release);
}

void Backend::worker_thread(int thread_id) {
    thread_local_id = thread_id; // 设置线程本地变量
    // TRACE_EVENT_BEGIN("schedule", "wait");
    bool first = true;
    uint64_t sleepy = 0;
    while (true) {
        ThreadStatus status =
            thread_state_[thread_id].status->load(std::memory_order_acquire);
        if (status == ThreadStatus::WORKING) {
            sleepy = 0;
            if (!first)
                TRACE_EVENT_END("schedule");
            process_tasks(thread_id);
            TRACE_EVENT_BEGIN("schedule", "wait");
            first = false;
        } else if (status == ThreadStatus::WAITING) {
            sleepy += 1;
            if (sleepy >= 4400000000) { // 4.4GHz
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            TRACE_EVENT_END("schedule");
            return;
        }
    }
}

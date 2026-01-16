/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:38
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_BACKEND_H
#define CPUINFER_BACKEND_H

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "perfetto/categories.h"

namespace heyi {
enum ThreadStatus {
    WORKING,
    WAITING,
    EXIT,
};

struct ThreadState {
    std::unique_ptr<std::atomic<ThreadStatus>> status;
    std::unique_ptr<std::atomic<int>> curr;
    int end;
};

class Backend {
  public:
    Backend(int);
    ~Backend();
    void start_trace(std::string file);
    void end_trace();
    int get_thread_num();
    void do_work_stealing_job(int, std::function<void(int)>,
                              std::function<void(int)>,
                              std::function<void(int)>);
    #ifdef USE_NUMA
    static thread_local int numa_node;
    #endif
    static thread_local int thread_local_id;

    FILE* tracing_file_;
    perfetto::TraceConfig tracing_config_;
    std::unique_ptr<perfetto::TracingSession> tracing_session_;

    std::vector<std::atomic<int>> input_conv_syn; // [nth]
    std::vector<std::vector<std::atomic<int>>> interm_conv_grp_syn; // [experts, nth]
    bool one_shot_mode;

  private:
    int thread_num_;
    int max_thread_num_;
    std::vector<ThreadState> thread_state_; // [thread_num]
    std::function<void(int)> init_func_;
    std::function<void(int)> compute_func_;
    std::function<void(int)> finalize_func_;
    std::vector<std::thread> workers_;

    void process_tasks(int);
    void worker_thread(int);
};

} // namespace heyi

#endif
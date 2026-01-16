/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"
#include "perfetto/categories.h"

using namespace heyi;

TaskQueue::TaskQueue(int max_task_num) {
    worker = std::thread(&TaskQueue::processTasks, this);
    sync_flags = std::vector<std::atomic<bool>>(max_task_num);
    for (auto &sync_flag : sync_flags) {
        sync_flag.store(true, std::memory_order_seq_cst);
    }
    exit_flag.store(false, std::memory_order_seq_cst);
}

TaskQueue::~TaskQueue() {
    {
        mutex.lock();
        exit_flag.store(true, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(int task_id, std::function<void()> task) {
    TRACE_EVENT_BEGIN("taskqueue", "enque");
    {
        mutex.lock();
        tasks.push({task_id, task});
        sync_flags.at(task_id).store(false, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_one();
}

void TaskQueue::sync(int task_id) {
    uint64_t sleepy = 0;
    while (!sync_flags.at(task_id).load(std::memory_order_seq_cst)) {
        sleepy += 1;
        if (sleepy >= 4400000000) { // 4.4GHz
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    TRACE_EVENT_END("taskqueue");
}

void TaskQueue::processTasks() {
    while (true) {
        std::function<void()> task;
        int task_id;
        {
            mutex.lock();
            cv.wait(mutex, [this]() { return !tasks.empty() || exit_flag.load(std::memory_order_seq_cst); });
            if (exit_flag.load(std::memory_order_seq_cst) && tasks.empty()) {
                return;
            }
            auto& [task_id_, task_] = tasks.front();
            task = std::move(task_);
            task_id = task_id_;
            tasks.pop();
            mutex.unlock();
        }
        task();
        {
            mutex.lock();
            sync_flags.at(task_id).store(true, std::memory_order_seq_cst);
            mutex.unlock();
        }
    }
}
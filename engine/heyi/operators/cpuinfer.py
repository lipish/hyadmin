#!/usr/bin/env python
# coding=utf-8
"""
Description  : This script defines the `CPUInferKVCache` and `CPUInfer` classes for performing inference 
               with a Key-Value Cache on the CPU. The `CPUInferKVCache` class is responsible for configuring 
               and managing key-value caches, updating and retrieving cache data, and handling attention 
               operations. It supports different cache types (e.g., Q4_0, FP16) and retrieval strategies 
               (e.g., shared, separate). The `CPUInfer` class handles task submission and synchronization 
               on the CPU, with optional CUDA stream integration for tasks involving GPU acceleration. 
               These classes facilitate efficient caching and memory management for deep learning models 
               that leverage key-value attention mechanisms, particularly on CPU-based systems.
Author       : djw
Date         : 2024-08-26 23:25:24
Version      : 1.0.0
LastEditors  : djw 
LastEditTime : 2024-08-26 23:25:24
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
"""
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build", "Debug"))
import heyi._ext as cpuinfer_ext


class CPUInfer:
    cpuinfer = None
    cur_backend_thread_num = 0
    cur_max_task_num = 0
    
    def __init__(self, thread_num, max_task_num):
        if thread_num != CPUInfer.cur_backend_thread_num or \
            max_task_num != CPUInfer.cur_max_task_num:
            CPUInfer.cur_backend_thread_num = thread_num
            CPUInfer.cur_max_task_num = max_task_num
            del CPUInfer.cpuinfer
            CPUInfer.cpuinfer = cpuinfer_ext.CPUInfer(thread_num, max_task_num)

    def submit(self, task):
        CPUInfer.cpuinfer.submit(task)

    def cuda_launch_host_func(self, current_cuda_stream, task):
        CPUInfer.cpuinfer.cuda_launch_host_func(current_cuda_stream, task)

    def sync(self, task_id):
        CPUInfer.cpuinfer.sync(task_id)

    # def sync_with_cuda_stream(self, task_id, current_cuda_stream):
    #     CPUInfer.cpuinfer.sync_with_cuda_stream(task_id, current_cuda_stream)

    def lock(self, stream):
        CPUInfer.cpuinfer.lock(stream)

    def unlock(self, stream):
        CPUInfer.cpuinfer.unlock(stream)


        

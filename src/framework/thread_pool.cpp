/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#include "framework/thread_pool.h"
#include "utils/logger.h"
#include <iostream>

namespace GryFlux
{

    ThreadPool::ThreadPool(size_t numThreads) : stop_(false)
    {
        // 确保至少有一个线程，或者使用系统硬件线程数
        if (numThreads == 0)
        {
            numThreads = std::thread::hardware_concurrency();
            // 至少一个线程
            if (numThreads == 0)
            {
                numThreads = 1;
            }
        }

        // 创建线程池中的工作线程
        for (size_t i = 0; i < numThreads; ++i)
        {
            workers_.emplace_back([this, i]
                                  {
            // 线程工作循环
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex_);
                    condition_.wait(lock, [this] { 
                        return stop_ || !tasks_.empty(); 
                    });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                // 执行任务
                try {
                    task();
                } catch (const std::exception& e) {
                    LOG.error("Exception in thread %zu: %s", i, e.what());
                } catch (...) {
                    LOG.error("Unknown exception in thread %zu", i);
                }
            } });
        }
        LOG.debug("[ThreadPool] Initialized with %zu threads", numThreads);
    }

    ThreadPool::~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            stop_ = true;
        }

        // 通知所有线程
        condition_.notify_all();

        // 等待所有线程完成
        for (std::thread &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        LOG.debug("[ThreadPool] Destroyed, all %zu threads joined", workers_.size());
    }

} // namespace GryFlux
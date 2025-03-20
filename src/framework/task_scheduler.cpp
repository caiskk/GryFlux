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
#include "framework/task_scheduler.h"
#include "utils/logger.h"
#include <iostream>
#include <unordered_set>
#include <queue>
#include <mutex>

namespace GryFlux
{
    // 添加互斥锁用于保护任务执行
    std::mutex taskExecutionMutex;

    TaskScheduler::TaskScheduler(size_t numThreads)
        : threadPool_(numThreads) {}

    void TaskScheduler::addTask(std::shared_ptr<TaskNode> task)
    {
        if (!task)
        {
            return;
        }
        tasks_[task->getId()] = task;
    }

    std::shared_ptr<TaskNode> TaskScheduler::getTask(const std::string &id)
    {
        auto it = tasks_.find(id);
        if (it != tasks_.end())
        {
            return it->second;
        }
        return nullptr;
    }

    std::shared_ptr<DataObject> TaskScheduler::execute(const std::string &outputTaskId)
    {
        auto outputTask = getTask(outputTaskId);
        if (!outputTask)
        {
            LOG.error("Task not found: %s", outputTaskId.c_str());
            return nullptr;
        }

        executeTask(outputTask);

        return outputTask->getResult();
    }

    void TaskScheduler::executeTask(std::shared_ptr<TaskNode> task)
    {
        // 防止空指针解引用
        if (!task) {
            LOG.error("Attempted to execute null task");
            return;
        }
        
        // 如果任务已经执行过，则直接返回
        if (task->isExecuted())
        {
            return;
        }

        // 首先执行所有依赖任务
        std::vector<std::future<void>> futures;
        for (const auto &dep : task->getDependencies())
        {
            if (dep && !dep->isExecuted())
            {
                futures.push_back(threadPool_.enqueue([this, dep]()
                {
                    try {
                        executeTask(dep);
                    } catch (const std::exception& e) {
                        LOG.error("Exception while executing dependency: %s", e.what());
                    } catch (...) {
                        LOG.error("Unknown exception while executing dependency");
                    }
                }));
            }
        }

        // 等待所有依赖任务完成
        for (auto &future : futures)
        {
            try {
                future.wait();
            } catch (const std::exception& e) {
                LOG.error("Exception while waiting for task dependency: %s", e.what());
            }
        }

        try {
            // 使用互斥锁保护任务执行过程
            std::lock_guard<std::mutex> lock(taskExecutionMutex);
            
            // 再次检查是否已执行，以防在等待依赖时被其他线程执行
            if (task->isExecuted()) {
                return;
            }

            // 记录任务开始执行时间
            task->startExecution();
            
            // 执行当前任务
            auto result = task->execute();
            
            // 记录任务结束执行时间
            task->endExecution();

            // 设置结果（必须在记录结束时间之后）
            task->setResult(result);
            
            // 记录执行时间
            LOG.debug("Task [%s] executed in %.3f ms", task->getId().c_str(), task->getExecutionTimeMs());
        } catch (const std::exception& e) {
            LOG.error("Exception in task [%s]: %s", task->getId().c_str(), e.what());
        } catch (...) {
            LOG.error("Unknown exception in task [%s]", task->getId().c_str());
        }
    }

    void TaskScheduler::clear()
    {
        tasks_.clear();
    }
    
    std::unordered_map<std::string, double> TaskScheduler::getTaskExecutionTimes() const
    {
        std::unordered_map<std::string, double> executionTimes;
        for (const auto& pair : tasks_)
        {
            if (pair.second->isExecuted())
            {
                executionTimes[pair.first] = pair.second->getExecutionTimeMs();
            }
        }
        return executionTimes;
    }

} // namespace GryFlux
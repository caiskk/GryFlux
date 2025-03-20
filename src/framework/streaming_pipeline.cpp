/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#include "framework/streaming_pipeline.h"
#include "utils/logger.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace GryFlux
{

    StreamingPipeline::StreamingPipeline(size_t numThreads, size_t queueSize)
        : pipelineBuilder_(std::make_shared<PipelineBuilder>(numThreads > 0 ? numThreads : std::thread::hardware_concurrency())),
          inputQueue_(std::make_shared<threadsafe_queue<std::shared_ptr<DataObject>>>()),
          outputQueue_(std::make_shared<threadsafe_queue<std::shared_ptr<DataObject>>>()),
          outputNodeId_("output"),
          running_(false),
          queueMaxSize_(queueSize),
          processedItems_(0),
          errorCount_(0),
          totalProcessingTime_(0),
          profilingEnabled_(false) {}

    StreamingPipeline::~StreamingPipeline()
    {
        stop();
    }

    void StreamingPipeline::start()
    {
        if (running_)
        {
            return;
        }

        if (!processor_)
        {
            throw std::runtime_error("Processor function not set");
        }

        // 重置统计数据
        processedItems_ = 0;
        errorCount_ = 0;
        totalProcessingTime_ = 0;
        taskStats_.clear(); // 重置任务统计数据
        startTime_ = std::chrono::high_resolution_clock::now();

        running_ = true;
        input_active_ = true;
        output_active_ = true;
        processingThread_ = std::thread(&StreamingPipeline::processingLoop, this);

        LOG.debug("[Pipeline] Started streaming pipeline");
    }

    void StreamingPipeline::stop()
    {
        if (!running_)
        {
            return;
        }

        running_ = false;
        input_active_ = false;

        if (processingThread_.joinable())
        {
            processingThread_.join();
        }

        output_active_ = false;

        // 清理PipelineBuilder对象
        pipelineBuilder_.reset();

        // 只有在启用性能分析时才输出统计数据
        if (profilingEnabled_)
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto totalTime = std::chrono::duration<double, std::milli>(endTime - startTime_).count();

            LOG.info("[Pipeline] Statistics:");
            LOG.info("  - Total items processed: %zu", processedItems_);
            LOG.info("  - Error count: %zu", errorCount_);
            LOG.info("  - Total running time: %.3f ms", totalTime);

            if (processedItems_ > 0)
            {
                double avgTime = static_cast<double>(totalProcessingTime_) / processedItems_;
                LOG.info("  - Average processing time per item: %.3f ms", avgTime);
                LOG.info("  - Processing rate: %.2f items/s", (processedItems_ * 1000.0 / totalTime));
            }

            // 输出同名任务的全局平均执行时间
            if (!taskStats_.empty())
            {
                LOG.info("[Pipeline] Global average execution time for tasks with the same name:");
                for (const auto &taskStat : taskStats_)
                {
                    const std::string &taskName = taskStat.first;
                    double totalTime = taskStat.second.first;
                    size_t count = taskStat.second.second;
                    double avgTime = totalTime / count;

                    LOG.info("  - Task [%s]: %.3f ms (average of %zu executions across all items)",
                             taskName.c_str(), avgTime, count);
                }
            }
        }
        else
        {
            LOG.debug("[Pipeline] Stopped streaming pipeline");
        }
    }

    void StreamingPipeline::setProcessor(ProcessorFunction processor)
    {
        if (running_)
        {
            throw std::runtime_error("Cannot set processor while pipeline is running");
        }
        processor_ = processor;
    }

    bool StreamingPipeline::addInput(std::shared_ptr<DataObject> data)
    {
        if (!data)
        {
            return false;
        }

        // 避免队列过大时的内存占用问题
        while (inputQueue_->size() >= queueMaxSize_ && input_active_.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (input_active_.load())
        {
            inputQueue_->push(data);
            return true;
        }
        return false;
    }

    bool StreamingPipeline::tryGetOutput(std::shared_ptr<DataObject> &output)
    {
        return outputQueue_->try_pop(output);
    }

    void StreamingPipeline::getOutput(std::shared_ptr<DataObject> &output)
    {
        outputQueue_->wait_and_pop(output);
    }

    void StreamingPipeline::setOutputNodeId(const std::string &outputId)
    {
        if (running_)
        {
            throw std::runtime_error("Cannot set output node ID while pipeline is running");
        }
        outputNodeId_ = outputId;
    }

    bool StreamingPipeline::inputEmpty() const
    {
        return inputQueue_->empty();
    }

    bool StreamingPipeline::outputEmpty() const
    {
        return outputQueue_->empty();
    }

    size_t StreamingPipeline::inputSize() const
    {
        return inputQueue_->size();
    }

    size_t StreamingPipeline::outputSize() const
    {
        return outputQueue_->size();
    }

    size_t StreamingPipeline::getProcessedItemCount() const
    {
        return processedItems_;
    }

    size_t StreamingPipeline::getErrorCount() const
    {
        return errorCount_;
    }

    bool StreamingPipeline::isRunning() const
    {
        return running_;
    }

    void StreamingPipeline::processingLoop()
    {
        while (running_ || !inputQueue_->empty())
        {
            std::shared_ptr<DataObject> input;
            if (inputQueue_->try_pop(input)) // 非阻塞获取输入
            {
                // 只有在启用性能分析时才测量时间
                std::chrono::time_point<std::chrono::high_resolution_clock> startProcess;
                if (profilingEnabled_)
                {
                    startProcess = std::chrono::high_resolution_clock::now();
                }

                try
                {
                    // 使用用户定义的处理器构建和执行管道
                    processor_(pipelineBuilder_, input, outputNodeId_);

                    // 只有在启用性能分析时才收集任务统计信息
                    if (profilingEnabled_)
                    {
                        auto result = pipelineBuilder_->execute(outputNodeId_);

                        // 收集同名任务的执行时间统计
                        auto taskTimes = pipelineBuilder_->getScheduler()->getTaskExecutionTimes();
                        for (const auto &taskTime : taskTimes)
                        {
                            const std::string &taskName = taskTime.first;
                            double executionTime = taskTime.second;

                            // 累加到同名任务的统计中
                            taskStats_[taskName].first += executionTime;
                            taskStats_[taskName].second++;
                        }

                        // 处理结果
                        if (result)
                        {
                            outputQueue_->push(result);
                            processedItems_++;
                        }

                        // 计算处理时间
                        auto endProcess = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration<double, std::milli>(endProcess - startProcess).count();
                        totalProcessingTime_ += duration;

                        LOG.debug("[Pipeline] Processed item %zu in %.3f ms", processedItems_, duration);
                    }
                    else
                    {
                        // 在不启用性能分析时，直接执行管道并处理结果
                        auto result = pipelineBuilder_->execute(outputNodeId_);
                        if (result)
                        {
                            outputQueue_->push(result);
                            processedItems_++;
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    errorCount_++;
                    LOG.error("[Pipeline] Error processing input: %s", e.what());
                }
                catch (...)
                {
                    errorCount_++;
                    LOG.error("[Pipeline] Unknown error processing input");
                }
            }
        }

        // 处理完所有输入后，关闭输出队列
        output_active_ = false;
        LOG.debug("[Pipeline] Processing loop completed");
    }

} // namespace GryFlux

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
#ifndef DATA_CONSUMER_H
#define DATA_CONSUMER_H

#include <atomic>
#include <thread>
#include <memory>
#include <functional>
#include "framework/streaming_pipeline.h"
#include "framework/data_object.h"
#include "utils/logger.h"
#include "utils/unified_allocator.h"

namespace GryFlux
{

    /**
     * 数据消费者基类 - 负责从StreamingPipeline接收输出数据并进行处理
     */
    class DataConsumer
    {
    protected:
        StreamingPipeline &pipeline;
        std::atomic<bool> &running;
        BaseUnifiedAllocator *allocator;
        std::thread consumer_thread;

    public:
        /**
         * 构造函数
         * @param pipeline 流处理管道
         * @param running 运行状态标志
         */
        DataConsumer(StreamingPipeline &pipeline, std::atomic<bool> &running, BaseUnifiedAllocator *allocator)
            : pipeline(pipeline), running(running), allocator(allocator), consumer_thread() {}

        /**
         * 析构函数
         */
        virtual ~DataConsumer()
        {
            stop();
        }

        /**
         * 启动消费者线程
         * @return 成功返回true，失败返回false
         */
        bool start()
        {
            try
            {
                consumer_thread = std::thread(&DataConsumer::run, this);
                return true;
            }
            catch (const std::exception &e)
            {
                LOG.error("[Consumer] Failed to start consumer thread: %s", e.what());
                return false;
            }
        }

        /**
         * 停止消费者线程
         */
        void stop()
        {
            running.store(false);
            if (consumer_thread.joinable())
            {
                consumer_thread.join();
            }
        }

        /**
         * 等待消费者线程结束
         */
        void join()
        {
            if (consumer_thread.joinable())
            {
                consumer_thread.join();
            }
        }

    protected:
        /**
         * 纯虚函数 - 具体的数据消费逻辑，需要被子类实现
         */
        virtual void run() = 0;

        /**
         * 从管道获取数据
         * @param data 接收数据的指针引用
         * @return 成功返回true，失败返回false
         */
        bool getData(std::shared_ptr<DataObject> &data)
        {
            return pipeline.tryGetOutput(data);
        }

        /**
         * 检查是否应该继续运行
         * @return 应该继续运行返回true，否则返回false
         */
        bool shouldContinue()
        {
            return running.load() || !pipeline.outputEmpty() || pipeline.isOutputActive();
        }
    };

} // namespace GryFlux

#endif // DATA_CONSUMER_H

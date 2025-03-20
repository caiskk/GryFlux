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
#ifndef DATA_PRODUCER_H
#define DATA_PRODUCER_H

#include <atomic>
#include <thread>
#include <memory>
#include "framework/streaming_pipeline.h"
#include "framework/data_object.h"
#include "utils/logger.h"
#include "utils/unified_allocator.h"

namespace GryFlux
{

    /**
     * 数据生产者基类 - 负责向StreamingPipeline提供输入数据
     */
    class DataProducer
    {
    protected:
        StreamingPipeline &pipeline;
        std::atomic<bool> &running;
        BaseUnifiedAllocator *allocator;
        std::thread producer_thread;

    public:
        /**
         * 构造函数
         * @param pipeline 流处理管道
         * @param running 运行状态标志
         */
        DataProducer(StreamingPipeline &pipeline, std::atomic<bool> &running, BaseUnifiedAllocator *allocator)
            : pipeline(pipeline), running(running), allocator(allocator), producer_thread() {}

        /**
         * 析构函数
         */
        virtual ~DataProducer()
        {
            stop();
        }

        /**
         * 启动生产者线程
         * @return 成功返回true，失败返回false
         */
        bool start()
        {
            try
            {
                producer_thread = std::thread(&DataProducer::run, this);
                return true;
            }
            catch (const std::exception &e)
            {
                LOG.error("[Producer] Failed to start producer thread: %s", e.what());
                return false;
            }
        }

        /**
         * 停止生产者线程
         */
        void stop()
        {
            running.store(false);
            pipeline.stop();
        }

        /**
         * 等待生产者线程结束
         */
        void join()
        {
            if (producer_thread.joinable())
            {
                producer_thread.join();
            }
        }

    protected:
        /**
         * 纯虚函数 - 具体的数据生产逻辑，需要被子类实现
         */
        virtual void run() = 0;

        /**
         * 向管道添加数据
         * @param data 数据对象
         * @return 成功返回true，失败返回false
         */
        bool addData(std::shared_ptr<DataObject> data)
        {
            if (!data)
            {
                LOG.warning("[Producer] Attempt to add null data");
                return false;
            }

            return pipeline.addInput(data);
        }
    };

} // namespace GryFlux

#endif // DATA_PRODUCER_H

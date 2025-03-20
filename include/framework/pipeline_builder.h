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
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "framework/task_scheduler.h"
#include "framework/task_node.h"
#include "framework/data_object.h"

namespace GryFlux
{

    // 流水线构建器
    class PipelineBuilder
    {
    public:
        PipelineBuilder(size_t numThreads = 0);

        // 添加输入数据源
        std::shared_ptr<TaskNode> addInput(const std::string &id, std::shared_ptr<DataObject> data);

        // 添加多输入处理节点
        std::shared_ptr<TaskNode> addTask(
            const std::string &id,
            std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)> func,
            const std::vector<std::shared_ptr<TaskNode>> &inputs);

        // 执行整个流水线，返回指定输出节点的结果
        std::shared_ptr<DataObject> execute(const std::string &outputId);

        // 重置流水线，以便重用
        void reset();

        // 设置是否启用性能分析
        void enableProfiling(bool enable) { profilingEnabled_ = enable; }

        // 获取性能分析状态
        bool isProfilingEnabled() const { return profilingEnabled_; }

        // 添加访问调度器的方法
        std::shared_ptr<TaskScheduler> getScheduler() const { return scheduler_; }

    private:
        std::shared_ptr<TaskScheduler> scheduler_;
        bool profilingEnabled_ = false;
    };

} // namespace GryFlux
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
#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>
#include "data_object.h"

namespace GryFlux
{

    /**
     * @brief 处理任务的基类
     * 所有计算节点任务都应该继承这个基类，并实现process方法
     */
    class ProcessingTask
    {
    public:
        ProcessingTask() {};
        virtual ~ProcessingTask() = default;

        /**
         * @brief 处理数据的核心方法
         * @param inputs 输入数据对象列表
         * @return 处理后的数据对象
         */
        virtual std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) = 0;

        /**
         * @brief 获取绑定到当前任务实例的函数对象
         * @return 处理函数
         */
        std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)>
        getProcessFunction()
        {
            return [this](const std::vector<std::shared_ptr<DataObject>> &inputs)
            {
                return this->process(inputs);
            };
        }
    };
    // 定义任务注册表类，用于管理所有处理任务
    class TaskRegistry
    {
    private:
        std::unordered_map<std::string, std::shared_ptr<ProcessingTask>> tasks;

    public:
        // 注册任务并返回任务ID
        template <typename T, typename... Args>
        std::string registerTask(const std::string &taskId, Args &&...args)
        {
            tasks[taskId] = std::make_shared<T>(std::forward<Args>(args)...);
            return taskId;
        }

        // 获取任务处理函数
        std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)>
        getProcessFunction(const std::string &taskId)
        {
            if (tasks.find(taskId) == tasks.end())
            {
                throw std::runtime_error("Task not found: " + taskId);
            }

            return [task = tasks[taskId]](const std::vector<std::shared_ptr<DataObject>> &inputs)
            {
                return task->process(inputs);
            };
        }
    };
} // namespace GryFlux

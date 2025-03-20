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
#include "test_producer.h"
#include "custom_package.h"

namespace GryFlux
{
    void TestImageProducer::run()
    {
        LOG.info("[TestImageProducer] Producer start");

        // 模拟从摄像头或其他源获取图像
        for (int i = 0; i < max_frames && running.load(); ++i)
        {
            // 生成测试图像
            auto input_data = std::make_shared<CustomPackage>();
            input_data->push_data(i);
            // 添加到处理管道
            if (!addData(input_data))
            {
                LOG.error("[TestImageProducer] Failed to add input data to pipeline");
                break;
            }

            frame_count++;

            // 模拟帧率
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_interval_ms));
        }
        this->stop();
        LOG.info("[TestImageProducer] Producer finished, generated %d frames", frame_count);
    }
};

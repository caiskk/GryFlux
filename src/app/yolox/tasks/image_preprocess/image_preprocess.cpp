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
#include "image_preprocess.h"
#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    std::shared_ptr<DataObject> ImagePreprocess::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // single input
        if (inputs.size() != 1) {
			return nullptr;
            
        }

        auto input_data = std::dynamic_pointer_cast<InputPackage>(inputs[0]);
        const auto& frame = input_data->get_data();
        cv::Mat img;
        //transform BGR -> RGB
        cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);
        // resize 
        auto img_width = input_data->get_width(), img_height = input_data->get_height();
        cv::Mat resized_img;
        int idx = input_data->get_id();
        float scale_w = 1, scale_h = 1; 
        if (img_width != model_width_ || img_height != model_height_) {
            cv::resize(img,resized_img,cv::Size(model_width_,model_height_));
            scale_w = (float)model_width_ / img_width, scale_h = (float)model_height_ / img_height;
        } else {
            resized_img = img;
        }

        LOG.info("get image id %d  img->width: %d, img->height: %d, scale_w: %f, scale_h: %f, resize img->width: %d, resize img->height: %d",
                 idx, img_width, img_height, scale_w, scale_h, resized_img.cols, resized_img.rows);
        // create new package
        return std::make_shared<InputPackage>(resized_img, idx, scale_w, scale_h);
    }
}

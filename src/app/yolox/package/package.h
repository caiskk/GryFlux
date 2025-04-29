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
#include <unordered_map>
#include <vector>

#include "framework/data_object.h"
#include "opencv2/opencv.hpp"


class InputPackage : public GryFlux::DataObject
{
public:

    InputPackage(const cv::Mat& frame, int idx, int scale_w = 1, int scale_h = 1)
    :src_frame_(frame), idx_(idx), scale_w_(scale_w), scale_h_(scale_h) {};
    ~InputPackage() {};

    const cv::Mat get_data() const {
      return src_frame_;
    }
    int get_id() const {
      return idx_;
    }
    int get_width() const {
      return src_frame_.cols;
    }
    int get_height() const {
      return src_frame_.rows;
    }
    float get_scale_w() const {
      return scale_w_;
    }
    float get_scale_h() const {
      return scale_h_;
    }
private:
  cv::Mat src_frame_;
  int idx_;
  float scale_w_;
  float scale_h_;
};

class OutputPackage : public GryFlux::DataObject
{
public:
    OutputPackage() {};
    ~OutputPackage() {};
    using OutputData = std::pair<std::shared_ptr<float[]>, std::size_t>; //buffer, size

    std::vector<OutputData> get_data() const {
      return rknn_output_buff;
    }
    void push_data(std::shared_ptr<float[]> data, std::size_t size) {
      rknn_output_buff.push_back({data, size});
    }
private:
    std::vector<OutputData> rknn_output_buff;
};
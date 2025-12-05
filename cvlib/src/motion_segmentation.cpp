/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{
    motion_segmentation::motion_segmentation(int frameQueueLength, int threshold){

        this->frameQueueLength = frameQueueLength;
        this->threshold = threshold;
        this->frameQueue = new std::deque<cv::Mat>;
    }

    motion_segmentation::~motion_segmentation(){
        if (frameQueue != nullptr){
            delete frameQueue;
        }
    }

    void motion_segmentation::setThreshold(int threshold){
        this->threshold = threshold;
    }

    void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double)
    {
        cv::Mat input_frame;
        cv::cvtColor(_image.getMat(), input_frame, cv::COLOR_BGR2GRAY);
        if (frameQueue->size() == 0){
            _fgmask.assign(cv::Mat::zeros(input_frame.size(), CV_8UC1));
            bg_model_ = cv::Mat::zeros(input_frame.size(), CV_8UC1);
            frameQueue->push_front(input_frame);
        }
        else{
            cv::Mat _sum = cv::Mat::zeros(input_frame.size(), CV_8UC1);
            for(const cv::Mat frame: *frameQueue){
                _sum += frame / frameQueue->size();
            }
            bg_model_ = _sum;
            frameQueue->push_front(input_frame);
            if (frameQueue->size() > frameQueueLength){
                frameQueue->pop_back();
            }
            _fgmask.assign(255 * (cv::abs(bg_model_ - input_frame) >= threshold));
        }
    }
} // namespace cvlib

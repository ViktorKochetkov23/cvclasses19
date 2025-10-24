/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2025-10-19
 * @author Viktor Kochetkov
 */


#include "cvlib.hpp"
#include <cmath>
#include <vector>

struct region
{
    int x;
    int y;
    int size_x;
    int size_y;
    double mean;
    double dev;
    region* bl;
    region* br;
    region* tl;
    region* tr;

    bool is_terminal(){
        return this->bl == nullptr;
    }

    region(int x, int y, int size_x, int size_y){
        this->x = x;
        this->y = y;
        this->size_x = size_x;
        this->size_y = size_y;
        this->bl = nullptr;
        this->br = nullptr;
        this->tl = nullptr;
        this->tr = nullptr;
    }

    ~region(){
        delete bl;
        delete br;
        delete tl;
        delete tr;
    }
};

namespace
{

std::tuple<double, double> stats_many(std::vector<region*> regions){
    double sum_many = 0.0;
    double sum_squared_many = 0.0;
    int size_many = 0;
    for (auto it=regions.begin(); it!=regions.end(); ++it){
        int size = ((*it)->size_x) * ((*it)->size_y);
        size_many += size;
        sum_many += (*it)->mean * size;
        sum_squared_many += ((*it)->dev) * ((*it)->dev) * size + size * ((*it)->mean) * ((*it)->mean);
    }
    double mean_many = sum_many / size_many;
    return std::make_tuple(mean_many, sqrt((sum_squared_many - 2*mean_many*sum_many + mean_many*mean_many) / size_many));
}

void split_image(cv::Mat image, double stddev, region* cur_region)
{
    if (image.empty())
    return;
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        cur_region->mean = mean.at<double>(0);
        cur_region->dev = dev.at<double>(0);
        // regions.push_back(cur_region);
        image.setTo(mean);
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    cur_region->tl = new region(cur_region->x, cur_region->y, height / 2, width / 2);
    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev, cur_region->tl);
    cur_region->bl = new region(cur_region->x, cur_region->y + width / 2, height / 2, width / 2);
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev, cur_region->bl);
    cur_region->br = new region(cur_region->x + height / 2, cur_region->y + width / 2, height / 2, width / 2);
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev, cur_region->br);
    cur_region->tr = new region(cur_region->x + height / 2, cur_region->y, height / 2, width / 2);
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev, cur_region->tr);
}

void merge_image(cv::Mat &image, double stddev, region* cur_region)
{

    if (cur_region->is_terminal()){
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    std::vector<region*> terminal_children;

    if (cur_region->bl->is_terminal()){
        terminal_children.push_back(cur_region->bl);
    }
    else{
        merge_image(image, stddev, cur_region->bl);
    }

    if (cur_region->br->is_terminal()){
        terminal_children.push_back(cur_region->br);
    }
    else{
        merge_image(image, stddev, cur_region->br);
    }

    if (cur_region->tl->is_terminal()){
        terminal_children.push_back(cur_region->tl);
    }
    else{
        merge_image(image, stddev, cur_region->tl);
    }

    if (cur_region->tr->is_terminal()){
        terminal_children.push_back(cur_region->tr);
    }
    else{
        merge_image(image, stddev, cur_region->tr);
    }


    std::tuple<double, double> stats = stats_many(terminal_children);

    if (std::get<1>(stats) < stddev){
        for (auto it=terminal_children.begin(); it!=terminal_children.end(); ++it){
            image(cv::Range((*it)->x, (*it)->x + (*it)->size_x), cv::Range((*it)->y, (*it)->y + (*it)->size_y)).setTo(std::get<0>(stats));
        }
    }
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    cv::Mat res = image;

    region* start_region = new region(0, 0, res.cols, res.rows);
    split_image(res, stddev, start_region);

    // merge part
    merge_image(res, stddev, start_region);
    delete start_region;
    return res;
}
} // namespace cvlib
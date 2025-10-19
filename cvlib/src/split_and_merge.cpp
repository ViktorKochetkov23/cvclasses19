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
};

namespace
{

int itersection_length(int left1, int right1, int left2, int right2){
    if (left1 > right2 || left2 > right1){
        return 0;
    }
    else{
        return std::min(right1,  right2) - std::max(left1, left2);
    }
}

void split_image(cv::Mat image, double stddev, std::vector<region> &regions, region cur_region)
{
    if (image.empty())
    return;
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        cur_region.mean = mean.at<double>(0);
        regions.push_back(cur_region);
        image.setTo(mean);
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev, regions, {.x = 0 + cur_region.x, .y = 0 + cur_region.y, .size_x = height / 2, .size_y = width / 2});
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev, regions, {.x = 0 + cur_region.x, .y = width / 2 + cur_region.y, .size_x = height / 2, .size_y = width / 2});
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev, regions, {.x = height / 2 + cur_region.x, .y = width / 2 + cur_region.y, .size_x = height / 2, .size_y = width / 2});
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev, regions, {.x = height / 2 + cur_region.x, .y = 0 + cur_region.y, .size_x = height / 2, .size_y = width / 2});
}

void merge_regions(cv::Mat &image, region &region1, region &region2, double stddev){
    int size1 = region1.size_x * region1.size_y;
    int size2 = region2.size_x * region2.size_y;
    double mean = (double)(size1 * region1.mean + size2*region2.mean)/(size1 + size2);
    double dev = std::sqrt((double)((region1.mean - mean) * (region1.mean - mean) * size1 + (region2.mean - mean) * (region2.mean - mean) * size2)/(size1 + size2));

    if (dev <= stddev)
    {
        image(cv::Range(region1.x, region1.x + region1.size_x), cv::Range(region1.y, region1.y + region1.size_y)).setTo(mean);
        image(cv::Range(region2.x, region2.x + region2.size_x), cv::Range(region2.y, region2.y + region2.size_y)).setTo(mean);
        region1.mean = mean;
        region2.mean = mean;
    }
}

void merge_image(cv::Mat &image, double stddev, std::vector<region> &regions)
{
    for(size_t i = 0; i < regions.size(); ++i)
    {
        if (regions[i].size_x == 0 || regions[i].size_y == 0){
            continue;
        }
        int x_remains = regions[i].size_x;
        int y_remains = regions[i].size_y;

        for(size_t j = i; (j < regions.size()); ++j)
        {
            if (regions[j].size_x != 0 && regions[j].size_y != 0){
                if (regions[i].x + regions[i].size_x == regions[j].x){
                    if (y_remains > 0 && (regions[i].y <= regions[j].y && regions[j].y <= regions[i].y + regions[i].size_y)){
                        merge_regions(image, regions[i], regions[j], stddev);
                        y_remains -= itersection_length(regions[i].y, regions[i].y + regions[i].size_y, regions[j].y, regions[j].y + regions[j].size_y);
                        continue;
                    }
                }
                if (regions[i].y + regions[i].size_y == regions[j].y){
                    if (x_remains > 0 && (regions[i].x <= regions[j].x && regions[j].x <= regions[i].x + regions[i].size_x)){
                        merge_regions(image, regions[i], regions[j], stddev);
                        x_remains -= itersection_length(regions[i].x, regions[i].x + regions[i].size_x, regions[j].x, regions[j].x + regions[j].size_x);
                    }
                }
            }
        }
    }
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    std::vector<region> p_regions;
    cv::Mat res = image;
    region start_region = {.x = 0, .y = 0, .size_x = res.cols, .size_y = res.rows};
    split_image(res, stddev, p_regions, start_region);

    // merge part
    merge_image(res, stddev, p_regions);
    return res;
}
} // namespace cvlib
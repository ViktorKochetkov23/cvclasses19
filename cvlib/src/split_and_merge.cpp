/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
void split_image(cv::Mat image, double stddev)
{
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev);
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev);
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev);
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev);
}

void merge_parts(cv::Mat image, double stddev)
{
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);
    printf("Mean %f\n", mean.at<double>(0));
    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    cv::Mat res = image;
    split_image(res, stddev);

    merge_parts(res, stddev);
    // \todo implement merge algorithm
    return res;
}
} // namespace cvlib

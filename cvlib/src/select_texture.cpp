/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

    double norm_l2() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::pow(std::abs(v), 2);
        }
        return std::sqrt(res);
    }
};

// Получить ядра фильтров Габора
void getGaborKernels(int kernel_size, std::vector<cv::Mat>& output){
    output.clear();
    const std::vector<double> th_values = {CV_PI / 3, 2 * CV_PI / 3};
    const std::vector<double> lm_values = {3, 4, 5};
    const std::vector<double> gm_values = {0.5, 0.8};
    const std::vector<double> psi_values = {0};
    for (const double& th : th_values){
        for (const double& lm: lm_values){
            for (const double& gm: gm_values){
                for (const double& psi: psi_values){
                    for (auto sig = 5; sig <= 15; sig += 5){
                        output.push_back(cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm, psi));
                    }
                }
            }
        }
    }
}

// Применить фильтры Габора ко всему изображению
void applyGaborFilters(const cv::Mat& image, int kernel_size, std::vector<cv::Mat>& output){
    output.clear();
    std::vector<cv::Mat> kernels;
    getGaborKernels(kernel_size, kernels);
    for (const cv::Mat& kernel: kernels){
        cv::Mat response;
        cv::filter2D(image, response, CV_32F, kernel);
        output.push_back(response);
    }
} 

// Получить дескриптор по roi и откликам фильтров Габора
void calculateDescriptor(const std::vector<cv::Mat>& responses, descriptor& descr, const cv::Rect& roi)
{
    descr.clear();
    cv::Mat mean, dev;
    for (const cv::Mat& response: responses){
        cv::meanStdDev(response(roi), mean, dev);
        descr.emplace_back(mean.at<double>(0));
        descr.emplace_back(dev.at<double>(0));
    }
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    const int kernel_size = 11;
    std::vector<cv::Mat> responses;
    applyGaborFilters(image, kernel_size, responses);
    descriptor reference;
    calculateDescriptor(responses, reference, roi);

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();

    // \todo move ROI smoothly pixel-by-pixel
    for (int i = 0; i < image.size().width  / roi.width; ++i)
    {
        for (int j = 0; j < image.size().height / roi.height; ++j)
        {
            auto curROI = baseROI + cv::Point(roi.width * i, roi.height * j);
            calculateDescriptor(responses, test, curROI);
            res(curROI) = 255 * ((reference - test).norm_l2() <= eps);
        }
    }

    return res;
}
} // namespace cvlib

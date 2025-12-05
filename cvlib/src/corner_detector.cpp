/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, 
                                  CV_OUT std::vector<cv::KeyPoint>& keypoints, 
                                  cv::InputArray)
{
    keypoints.clear();

    cv::Mat gray;
    cv::cvtColor(image.getMat(), gray, cv::COLOR_BGR2GRAY);

    const int threshold = 20;

    // Смещения пикселей круга (x, y)
    static const int offset_x[16] = {
         0,  1,  2,  3,
         3,  3,  2,  1,
         0, -1, -2, -3,
        -3, -3, -2, -1
    };

    static const int offset_y[16] = {
        -3, -3, -2, -1,
         0,  1,  2,  3,
         3,  3,  2,  1,
         0, -1, -2, -3
    };

    // Индексы 1,5,9,13
    static const int test_id[4] = {0, 4, 8, 12};

    const int rows = gray.rows;
    const int cols = gray.cols;

    for (int j = 4; j < rows - 4; ++j) {
        const uchar* row_ptr = gray.ptr<uchar>(j);

        for (int i = 4; i < cols - 4; ++i) {
            const uchar center = row_ptr[i];

            // Быстрая проверка 4 пикселей
            int passed = 0;
            for (int k = 0; k < 4; ++k) {
                int idx = test_id[k];
                int px = i + offset_x[idx];
                int py = j + offset_y[idx];

                uchar pix = gray.ptr<uchar>(py)[px];
                if (std::abs(pix - center) > threshold)
                    passed++;
            }
            if (passed < 3)
                continue;

            // Полная проверка 16 пикселей
            bool flag[16];
            for (int k = 0; k < 16; ++k) {
                int px = i + offset_x[k];
                int py = j + offset_y[k];

                uchar pix = gray.ptr<uchar>(py)[px];
                flag[k] = std::abs(pix - center) > threshold;
            }

            // Проверка 9 последовательных
            bool is_corner = false;
            for (int start = 0; start < 16 && !is_corner; ++start) {
                int c = 0;
                for (int t = 0; t < 16; ++t) {
                    int idx = (start + t) & 15; // быстрее чем %
                    if (flag[idx]) {
                        if (++c >= 9) {
                            is_corner = true;
                            break;
                        }
                    } else {
                        c = 0;
                    }
                }
            }

            if (is_corner) {
                keypoints.emplace_back(i, j, 7);
            }
        }
    }
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib

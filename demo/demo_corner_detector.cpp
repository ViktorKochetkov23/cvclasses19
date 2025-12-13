/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_corner_detector(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector = cvlib::corner_detector_fast::create(); // \todo use cvlib::corner_detector_fast
    std::vector<cv::KeyPoint> corners;

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);

        std::stringstream ss;
        ss << corners.size();;
        const auto txtFont = cv::FONT_HERSHEY_SIMPLEX;
        const auto fontScale = 0.5;
        const auto thickness = 1;
        static const cv::Size textSize = cv::getTextSize("fps: 19.127", txtFont, fontScale, thickness, nullptr);
        static const cv::Point textOrgPoint = {textSize.width / 2, 20};

        cv::putText(frame, ss.str(), textOrgPoint, txtFont, fontScale, {0, 255, 0}, thickness, 8, false);

        cv::imshow(demo_wnd, frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}

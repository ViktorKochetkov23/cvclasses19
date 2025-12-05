/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

void onThresholdChange(int value, void* param){
    cvlib::motion_segmentation* mseg = reinterpret_cast<cvlib::motion_segmentation*>(param);
    mseg->setThreshold(value);
}

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    // auto mseg = cv::createBackgroundSubtractorMOG2(); // \todo use cvlib::motion_segmentation
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    int threshold = 50;
    cvlib::motion_segmentation* mseg = new cvlib::motion_segmentation(10, threshold);
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::createTrackbar("th", demo_wnd, &threshold, 255, onThresholdChange, mseg);

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        // mseg->setVarThreshold(threshold); // \todo use TackbarCallback
        mseg->apply(frame, frame_mseg);
        if (!frame_mseg.empty())
            cv::imshow(demo_wnd, frame_mseg);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    delete mseg;
    return 0;
}

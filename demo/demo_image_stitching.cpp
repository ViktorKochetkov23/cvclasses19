/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    cvlib::Stitcher stitcher;
    // If you want to use your own detector / matcher, set them here:
    // cv::Ptr<cvlib::corner_detector_fast> myDetector = cvlib::corner_detector_fast::create();
    // cvlib::descriptor_matcher myMatcher = cvlib::descriptor_matcher(1.2f);
    // stitcher.setFeatureDetector(myDetector);
    // stitcher.setMatcher(cv::makePtr<cvlib::descriptor_matcher>(myMatcher));

    cv::Mat pano;
    int frame_index = 0;

    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    cv::namedWindow("Panorama", cv::WINDOW_NORMAL);

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Camera", frame);

        int key = cv::waitKey(10);
        if (key == 27) // ESC exit
            break;
        else if (key == 32) // Space pressed -> save and stitch
        {
            std::string fname = "frame_" + std::to_string(frame_index++) + ".jpg";
            cv::imwrite(fname, frame);
            std::cout << "Saved " << fname << std::endl;

            bool ok = stitcher.stitch(frame, pano);
            if (!ok)
            {
                std::cerr << "Stitch failed for this frame." << std::endl;
            }
            else
            {
                cv::imshow("Panorama", pano);
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

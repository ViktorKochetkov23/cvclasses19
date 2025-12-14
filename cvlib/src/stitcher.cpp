#include <opencv2/opencv.hpp>
#include "cvlib.hpp"

namespace cvlib{

void Stitcher::setFeatureDetector(cv::Ptr<cv::Feature2D> det){
    detector_ = det;
}

void Stitcher::setMatcher(cv::Ptr<cv::DescriptorMatcher> mat){
    matcher_ = mat;
}

void Stitcher::setRatioThreshold(float r) {
    ratio_thresh_ = r;
}

bool Stitcher::stitch(const cv::Mat& img, cv::Mat& pano) {
if (img.empty()) return false;

    if (pano.empty())
    {
        pano = img.clone();
        return true;
    }

    // 1) detect & compute
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat d1, d2;
    detector_->detectAndCompute(pano, cv::noArray(), kp1, d1);
    detector_->detectAndCompute(img, cv::noArray(), kp2, d2);

    if (d1.empty() || d2.empty())
    {
        std::cerr << "No descriptors found." << std::endl;
        return false;
    }

    // 2) knn match (2) and ratio test
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(d2, d1, knn, 2); // query = img, train = pano

    std::vector<cv::DMatch> good;
    for (auto &m : knn)
    {
        good.push_back(m[0]);
    }

    if (good.size() < 4)
    {
        std::cerr << "Not enough good matches: " << good.size() << std::endl;
        return false;
    }

    // 3) Prepare points for homography: img -> pano
    std::vector<cv::Point2f> pts_img, pts_pano;
    for (auto &m : good)
    {
        pts_img.push_back(kp2[m.queryIdx].pt);
        pts_pano.push_back(kp1[m.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(pts_img, pts_pano, cv::RANSAC, 3.0, mask);
    if (H.empty())
    {
        std::cerr << "Homography estimation failed." << std::endl;
        return false;
    }

    // 4) compute size of resulting panorama by transforming corners
    std::vector<cv::Point2f> corners_img(4), corners_pano(4), corners_img_tr(4);
    corners_img[0] = cv::Point2f(0,0);
    corners_img[1] = cv::Point2f((float)img.cols,0);
    corners_img[2] = cv::Point2f((float)img.cols,(float)img.rows);
    corners_img[3] = cv::Point2f(0,(float)img.rows);

    corners_pano[0] = cv::Point2f(0,0);
    corners_pano[1] = cv::Point2f((float)pano.cols,0);
    corners_pano[2] = cv::Point2f((float)pano.cols,(float)pano.rows);
    corners_pano[3] = cv::Point2f(0,(float)pano.rows);

    cv::perspectiveTransform(corners_img, corners_img_tr, H);

    // union box
    float min_x = std::min( min_element_value(corners_img_tr, true), 0.0f );
    float min_y = std::min( min_element_value(corners_img_tr, false), 0.0f );
    min_x = std::min(min_x, 0.0f);
    min_y = std::min(min_y, 0.0f);
    float max_x = std::max( max_element_value(corners_img_tr, true), (float)pano.cols );
    float max_y = std::max( max_element_value(corners_img_tr, false), (float)pano.rows );

    // also consider pano corners explicitly
    max_x = std::max(max_x, max_element_value(corners_pano, true));
    max_y = std::max(max_y, max_element_value(corners_pano, false));
    min_x = std::min(min_x, min_element_value(corners_pano, true));
    min_y = std::min(min_y, min_element_value(corners_pano, false));

    int width = static_cast<int>(std::ceil(max_x - min_x));
    int height = static_cast<int>(std::ceil(max_y - min_y));
    if (width <= 0 || height <= 0)
    {
        std::cerr << "Invalid panorama size." << std::endl;
        return false;
    }

    // 5) translation to keep everything positive
    cv::Mat T = (cv::Mat_<double>(3,3) << 1,0,-min_x, 0,1,-min_y, 0,0,1);
    cv::Mat Ht = T * H;

    // 6) warp img into result
    cv::Mat result(height, width, pano.type(), cv::Scalar::all(0));
    cv::warpPerspective(img, result, Ht, result.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    // 7) copy pano into result with offset (-min_x, -min_y)
    int offset_x = static_cast<int>(-min_x);
    int offset_y = static_cast<int>(-min_y);

    // create mask where pano exists
    cv::Mat mask_pano(pano.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < pano.rows; ++y)
        for (int x = 0; x < pano.cols; ++x)
        {
            // simple non-black test (works for typical images)
            if (pano.type() == CV_8UC3)
            {
                cv::Vec3b v = pano.at<cv::Vec3b>(y,x);
                if (v != cv::Vec3b(0,0,0)) mask_pano.at<unsigned char>(y,x) = 255;
            }
            else
            {
                if (pano.at<unsigned char>(y,x) != 0) mask_pano.at<unsigned char>(y,x) = 255;
            }
        }

    // place pano with simple blending: where both exist -> average
    for (int y = 0; y < pano.rows; ++y)
    {
        int ry = y + offset_y;
        if (ry < 0 || ry >= result.rows) continue;
        for (int x = 0; x < pano.cols; ++x)
        {
            int rx = x + offset_x;
            if (rx < 0 || rx >= result.cols) continue;

            if (pano.type() == CV_8UC3)
            {
                cv::Vec3b src = pano.at<cv::Vec3b>(y,x);
                cv::Vec3b dst = result.at<cv::Vec3b>(ry,rx);
                bool src_nonzero = (src != cv::Vec3b(0,0,0));
                bool dst_nonzero = (dst != cv::Vec3b(0,0,0));

                if (dst_nonzero && src_nonzero)
                {
                    // simple average
                    cv::Vec3b res;
                    for (int c = 0; c < 3; ++c) res[c] = (uchar)((int)src[c] / 2 + (int)dst[c] / 2);
                    result.at<cv::Vec3b>(ry,rx) = res;
                }
                else if (src_nonzero)
                {
                    result.at<cv::Vec3b>(ry,rx) = src;
                }
                // else keep warped pixel (dst)
            }
            else
            {
                uchar src = pano.at<uchar>(y,x);
                uchar dst = result.at<uchar>(ry,rx);
                if (src && dst)
                    result.at<uchar>(ry,rx) = (src + dst) / 2;
                else if (src)
                    result.at<uchar>(ry,rx) = src;
            }
        }
    }

    pano = result; // update pano
    return true;
}

float Stitcher::min_element_value(const std::vector<cv::Point2f>& pts, bool x){
    float v = (x ? pts[0].x : pts[0].y);
    for (size_t i = 1; i < pts.size(); ++i) v = std::min(v, (x ? pts[i].x : pts[i].y));
    return v;
}

float Stitcher::max_element_value(const std::vector<cv::Point2f>& pts, bool x)
{
    float v = (x ? pts[0].x : pts[0].y);
    for (size_t i = 1; i < pts.size(); ++i) v = std::max(v, (x ? pts[i].x : pts[i].y));
    return v;
}
}
/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
float descriptor_matcher::ssd(const cv::Mat& a, const cv::Mat& b)
{
    float sum = 0.f;
    for (int i = 0; i < a.cols; ++i)
    {
        float d = a.at<float>(0, i) - b.at<float>(0, i);
        sum += d * d;
    }
    return sum;
}

void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.resize(q_desc.rows);

    cv::RNG rnd;
    for (int i = 0; i < q_desc.rows; ++i)
    {
        // \todo implement Ratio of SSD check.
        matches[i].emplace_back(i, rnd.uniform(0, t_desc.rows), FLT_MAX);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    // \todo implement matching with "maxDistance"
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.clear();
    matches.resize(q_desc.rows);

    for (int i = 0; i < q_desc.rows; ++i)
    {
        const cv::Mat q = q_desc.row(i);

        int best_id = -1;
        float best = std::numeric_limits<float>::max();;
        float second_best = std::numeric_limits<float>::max();

        for (int j = 0; j < t_desc.rows; ++j)
        {
            const cv::Mat t = t_desc.row(j);

            float dist = ssd(q, t);
            if (best > dist) {
                second_best = best;
                best = dist;
                best_id = j;
            }
            else if (second_best > dist){
                second_best = dist;
            }
        }

        if (best_id != -1){
            if (best < maxDistance && (best / second_best) < ratio_){
                matches[0].emplace_back(i, best_id, best);
            }
        }
    }
    // knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib

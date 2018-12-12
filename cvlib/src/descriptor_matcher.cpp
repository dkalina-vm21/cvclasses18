/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Darina Kalina
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
	for (auto& match : matches)
	{
		if (match.size() > 1)
		{
			std::set<cv::DMatch> true_matches;

			for (int i = 0; i < match.size() - 1; ++i)
			{
				for (int j = i + 1; j < match.size(); ++j)
				{
					auto min_dist = std::min(match[i].distance, match[j].distance);

					if (min_dist == match[i].distance &&
						match[i].distance / match[j].distance < ratio_)
						true_matches.insert(match[i]);

					else if (min_dist == match[j].distance &&
						match[j].distance / match[i].distance < ratio_)
						true_matches.insert(match[j]);
				}
			}

			std::copy(true_matches.begin(), true_matches.end(), match.begin());
			match.erase(match.begin() + true_matches.size(), match.end());
		}
	}
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
	if (trainDescCollection.empty())
		return;

	auto q_desc = queryDescriptors.getMat();
	auto& t_desc = trainDescCollection[0];

	matches.resize(q_desc.rows);

	for (int i = 0; i < q_desc.rows; ++i)
	{
		for (auto j = 0; j < t_desc.rows; ++j)
		{
			double ssd_dist = cv::norm(q_desc.row(i) - t_desc.row(j), cv::NORM_L2SQR);
			if (ssd_dist < maxDistance)
				matches[i].emplace_back(i, j, ssd_dist);
		}
	}

    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib

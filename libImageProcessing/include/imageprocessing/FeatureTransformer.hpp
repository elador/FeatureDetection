/*
 * FeatureTransformer.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef FEATURETRANSFORMER_HPP_
#define FEATURETRANSFORMER_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace imageprocessing {

/**
 * Feature transformer that transforms an image patch to a feature vector.
 */
class FeatureTransformer {
public:

	virtual ~FeatureTransformer() {}

	/**
	 * Extracts the feature vector based on an image patch.
	 *
	 * @param[in] patch The image patch.
	 * @return A row vector containing the feature values.
	 */
	virtual Mat transform(const Mat& patch) = 0;
};

} /* namespace imageprocessing */
#endif /* FEATURETRANSFORMER_HPP_ */

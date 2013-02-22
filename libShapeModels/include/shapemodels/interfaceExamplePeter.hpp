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
 * Feature transformer that transforms an image patch into a feature vector. That feature vector should be a row vector,
 * but in some special cases that might not be the case.
 */
class FeatureTransformer {
public:

	virtual ~FeatureTransformer() {}

	/**
	 * Transforms the given image patch into a feature vector.
	 *
	 * @param[in,out] patch The image patch that is transformed into a feature vector.
	 */
	virtual void transform(Mat& patch) const = 0;
};

} /* namespace imageprocessing */
#endif /* FEATURETRANSFORMER_HPP_ */

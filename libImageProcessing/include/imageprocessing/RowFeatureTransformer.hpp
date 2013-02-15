/*
 * RowFeatureTransformer.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef ROWFEATURETRANSFORMER_HPP_
#define ROWFEATURETRANSFORMER_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace imageprocessing {

/**
 * Feature transformer that applies image filters to an image patch and transforms it to a row vector.
 */
class RowFeatureTransformer {
public:

	explicit RowFeatureTransformer();

	~RowFeatureTransformer();

	Mat transform(const Mat& patch);

	// TODO liste von ImageFilter -> zuerst filtern, dann transformieren (aneinanderhängen)
};

} /* namespace imageprocessing */
#endif /* ROWFEATURETRANSFORMER_HPP_ */

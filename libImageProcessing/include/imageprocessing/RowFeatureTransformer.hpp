/*
 * RowFeatureTransformer.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef ROWFEATURETRANSFORMER_HPP_
#define ROWFEATURETRANSFORMER_HPP_

#include "imageprocessing/FeatureTransformer.hpp"

namespace imageprocessing {

/**
 * Feature transformer that applies image filters to an image patch and transforms it to a row vector.
 */
class RowFeatureTransformer : public FeatureTransformer {
public:

	/**
	 * Constructs a new row feature transformer.
	 */
	explicit RowFeatureTransformer() {}

	~RowFeatureTransformer() {}

	void transform(Mat& patch) const {
		patch = patch.reshape(0, 1);
	}
};

} /* namespace imageprocessing */
#endif /* ROWFEATURETRANSFORMER_HPP_ */

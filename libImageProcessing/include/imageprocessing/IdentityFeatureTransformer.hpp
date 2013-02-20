/*
 * IdentityFeatureTransformer.hpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#ifndef IDENTITYFEATURETRANSFORMER_HPP_
#define IDENTITYFEATURETRANSFORMER_HPP_

#include "imageprocessing/FeatureTransformer.hpp"

namespace imageprocessing {

/**
 * Feature transformer that leaves the patch data as it is (and therefore the result might not be a row vector).
 */
class IdentityFeatureTransformer : public FeatureTransformer {
public:

	/**
	 * Constructs a new identity feature transformer.
	 */
	explicit IdentityFeatureTransformer() {}

	~IdentityFeatureTransformer() {}

	void transform(Mat& patch) const {}
};

} /* namespace imageprocessing */
#endif /* IDENTITYFEATURETRANSFORMER_HPP_ */

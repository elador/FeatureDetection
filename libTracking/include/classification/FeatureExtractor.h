/*
 * FeatureExtractor.h
 *
 *  Created on: 19.11.2012
 *      Author: poschmann
 */

#ifndef FEATUREEXTRACTOR_H_
#define FEATUREEXTRACTOR_H_

#include "opencv2/core/core.hpp"
#include <memory>

using std::shared_ptr;

namespace classification {

class FeatureVector;

/**
 * Feature extractor that constructs a feature vector given an image and a sample.
 */
class FeatureExtractor {
public:

	virtual ~FeatureExtractor() {}

	/**
	 * Initializes this feature extractor with a new image.
	 *
	 * @param[in] image The image.
	 */
	virtual void init(const cv::Mat& image) = 0;

	/**
	 * Extracts the feature vector at a certain location (patch) of the current image.
	 *
	 * @param[in] x The x-coordinate of the center position of the patch.
	 * @param[in] y The y-coordinate of the center position of the patch.
	 * @param[in] size The size (e.g. width, height, diameter) of the patch.
	 * @return A pointer to the feature vector that might be empty if no feature vector could be created.
	 */
	virtual shared_ptr<FeatureVector> extract(int x, int y, int size) = 0;
};

} /* namespace classification */
#endif /* FEATUREEXTRACTOR_H_ */

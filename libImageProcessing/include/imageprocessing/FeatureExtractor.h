/*
 * FeatureExtractor.h
 *
 *  Created on: 19.11.2012
 *      Author: poschmann
 */

#ifndef FEATUREEXTRACTOR_H_
#define FEATUREEXTRACTOR_H_

#include "imageprocessing/PatchExtractor.hpp"
#include "imageprocessing/FeatureTransformer.hpp"
#include "opencv2/core/core.hpp"
#include <memory>

using cv::Mat;
using std::shared_ptr;

namespace imageprocessing {

/**
 * Feature extractor that constructs a feature vector given an image and a rectangular size.
 */
class FeatureExtractor {
public:

	/**
	 * Blah. TODO parameter
	 */
	FeatureExtractor();

	~FeatureExtractor();

	/**
	 * Updates this feature extractor with a new image.
	 *
	 * @param[in] image The image.
	 */
	void update(const Mat& image);

	/**
	 * Extracts the feature vector at a certain location (patch) of the current image.
	 *
	 * @param[in] x The x-coordinate of the center position of the patch.
	 * @param[in] y The y-coordinate of the center position of the patch.
	 * @param[in] size The size (e.g. width, height, diameter) of the patch.
	 * @return A pointer to the feature vector that might be empty if no feature vector could be created.
	 */
	Mat extract(int x, int y, int size);

private:

	shared_ptr<PatchExtractor> extractor;       ///< The patch extractor.
	shared_ptr<RowFeatureTransformer> transformer; ///< The feature transformer.
};

} /* namespace imageprocessing */
#endif /* FEATUREEXTRACTOR_H_ */

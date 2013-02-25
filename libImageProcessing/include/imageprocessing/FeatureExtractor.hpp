/*
 * FeatureExtractor.hpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#ifndef FEATUREEXTRACTOR_HPP_
#define FEATUREEXTRACTOR_HPP_

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
	 * Constructs a new feature extractor.
	 *
	 * @param[in] extractor The patch extractor.
	 * @param[in] transformer The feature transformer.
	 */
	explicit FeatureExtractor(shared_ptr<PatchExtractor> extractor, shared_ptr<FeatureTransformer> transformer) :
		extractor(extractor), transformer(transformer) {}

	~FeatureExtractor() {}

	/**
	 * Updates this feature extractor, so subsequent extracted features are based on the new image.
	 *
	 * @param[in] image The new source image of extracted features.
	 */
	void update(const Mat& image) {
		extractor->update(image);
	}

	/**
	 * Extracts the feature vector at a certain location (patch) of the current image.
	 *
	 * @param[in] x The x-coordinate of the center position of the patch.
	 * @param[in] y The y-coordinate of the center position of the patch.
	 * @param[in] width The width of the patch.
	 * @param[in] height The height of the patch.
	 * @return A pointer to the patch (with its feature vector) that might be empty if the patch could not be created.
	 */
	shared_ptr<Patch> extract(int x, int y, int width, int height) {
		shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
		if (!patch)
			return Mat;		// Patrik: TODO There's a compiler error here (return-type)
		transformer->transform(patch->getData());
		return patch;
	}

private:

	shared_ptr<PatchExtractor> extractor;       ///< The patch extractor.
	shared_ptr<FeatureTransformer> transformer; ///< The feature transformer.
};

} /* namespace imageprocessing */
#endif /* FEATUREEXTRACTOR_HPP_ */

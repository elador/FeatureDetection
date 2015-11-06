/*
 * PatchResizingFeatureExtractor.hpp
 *
 *  Created on: 24.05.2013
 *      Author: poschmann
 */

#ifndef PATCHRESIZINGFEATUREEXTRACTOR_HPP_
#define PATCHRESIZINGFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"

namespace imageprocessing {

/**
 * Feature extractor that manipulates the positioning and size of the patch before extraction, so it may focus on the
 * center of the original patch or capture the surroundings. The position and size of the extracted patch are then
 * changed, so they match the original extract request and further processing works with the original patch position
 * and size. The patch (image) data will remain unchanged.
 */
class PatchResizingFeatureExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new patch resizing feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor.
	 * @param[in] factor The patch resizing factor.
	 * @param[in] yOffset The y-offset relative to the original patch height.
	 * @param[in] xOffset The x-offset relative to the original patch width.
	 */
	PatchResizingFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor, double factor, double yOffset = 0, double xOffset = 0) :
			extractor(extractor), factor(factor), yOffset(yOffset), xOffset(xOffset) {}

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image) {
		extractor->update(image);
	}

	std::shared_ptr<Patch> extract(int x, int y, int width, int height) const {
		std::shared_ptr<Patch> patch = extractor->extract(
				cvRound(x + xOffset * width), cvRound(y + yOffset * height),
				cvRound(factor * width), cvRound(factor * height));
		if (patch) {
			patch->setWidth(cvRound(patch->getWidth() / factor));
			patch->setHeight(cvRound(patch->getHeight() / factor));
			patch->setX(cvRound(patch->getX() - xOffset * patch->getWidth()));
			patch->setY(cvRound(patch->getY() - yOffset * patch->getHeight()));
		}
		return patch;
	}

private:

	std::shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	double factor;  ///< The patch resizing factor.
	double yOffset; ///< The y-offset relative to the original patch height.
	double xOffset; ///< The x-offset relative to the original patch height.
};

} /* namespace imageprocessing */
#endif /* PATCHRESIZINGFEATUREEXTRACTOR_HPP_ */

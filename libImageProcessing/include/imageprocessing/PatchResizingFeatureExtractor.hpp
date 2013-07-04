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
 * Feature extractor that extracts surroundings of the requested patch or focusses on the patch center. It scales the
 * patch size prior to extracting the patch from the underlying feature extractor and corrects the size of the patch
 * afterwards (reverses the scaling), so further processing works with the original patch size. Additionally, the patch
 * center can be moved by an offset.
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
	PatchResizingFeatureExtractor(shared_ptr<FeatureExtractor> extractor, double factor, double yOffset = 0, double xOffset = 0) :
			extractor(extractor), factor(factor), yOffset(yOffset), xOffset(xOffset) {}

	~PatchResizingFeatureExtractor() {}

	void update(const Mat& image) {
		extractor->update(image);
	}

	void update(shared_ptr<VersionedImage> image) {
		extractor->update(image);
	}

	shared_ptr<Patch> extract(int x, int y, int width, int height) const {
		shared_ptr<Patch> patch = extractor->extract(
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

	shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	double factor;                          ///< The patch resizing factor.
	double yOffset;                         ///< The y-offset relative to the original patch height.
	double xOffset;                         ///< The x-offset relative to the original patch height.
};

} /* namespace imageprocessing */
#endif /* PATCHRESIZINGFEATUREEXTRACTOR_HPP_ */

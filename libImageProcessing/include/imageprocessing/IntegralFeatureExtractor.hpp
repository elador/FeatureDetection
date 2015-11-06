/*
 * IntegralFeatureExtractor.hpp
 *
 *  Created on: 06.08.2013
 *      Author: poschmann
 */

#ifndef INTEGRALFEATUREEXTRACTOR_HPP_
#define INTEGRALFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"

namespace imageprocessing {

/**
 * Feature extractor based on integral images. Because integral images are one pixel wider and higher, the size
 * of the patch request is increased by one to include another row and column to the bottom and right of the patch.
 */
class IntegralFeatureExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new integral feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor.
	 */
	explicit IntegralFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor) : extractor(extractor) {}

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image) {
		extractor->update(image);
	}

	std::shared_ptr<Patch> extract(int x, int y, int width, int height) const {
		int offsetX = width % 2 == 0 ? 0 : 1;
		int offsetY = height % 2 == 0 ? 0 : 1;
		std::shared_ptr<Patch> patch = extractor->extract(x + offsetX, y + offsetY, width + 1, height + 1);
		if (patch) {
			patch->setX(patch->getX() - offsetX);
			patch->setY(patch->getY() - offsetY);
			patch->setWidth(patch->getWidth() - 1);
			patch->setHeight(patch->getHeight() - 1);
		}
		return patch;
	}

private:

	std::shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
};

} /* namespace imageprocessing */
#endif /* INTEGRALFEATUREEXTRACTOR_HPP_ */

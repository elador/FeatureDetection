/*
 * PatchExtractor.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef PATCHEXTRACTOR_HPP_
#define PATCHEXTRACTOR_HPP_

#include "opencv2/core/core.hpp"
#include <memory>

using cv::Mat;
using std::shared_ptr;

namespace imageprocessing {

class Patch;

/**
 * Extracts patches from an image.
 */
class PatchExtractor {
public:

	virtual ~PatchExtractor() {}

	/**
	 * Updates this extractor, so subsequent extracted patches are based on the new image.
	 *
	 * @param[in] image The new source image of extracted patches.
	 */
	virtual void update(const Mat& image) = 0;

	/**
	 * Extracts a patch from the corresponding image.
	 *
	 * @param[in] x The x-coordinate of the patch center in the original image.
	 * @param[in] y The y-coordinate of the patch center in the original image.
	 * @param[in] width The width of the patch in the image.
	 * @param[in] height The height of the patch in the image.
	 * @return The extracted patch (might have a size other than the given one).
	 */
	virtual shared_ptr<Patch> extract(int x, int y, int width, int height) const = 0;
};

} /* namespace imageprocessing */
#endif /* PATCHEXTRACTOR_HPP_ */

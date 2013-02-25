/*
 * PyramidPatchExtractor.hpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#ifndef PYRAMIDPATCHEXTRACTOR_HPP_
#define PYRAMIDPATCHEXTRACTOR_HPP_

#include "imageprocessing/PatchExtractor.hpp"
#include <vector>

using std::vector;

namespace imageprocessing {

class ImagePyramid;

/**
 * Extracts patches of a constant size from an image pyramid.
 */
class PyramidPatchExtractor : public PatchExtractor {
public:

	/**
	 * Constructs a new pyramid based patch extractor.
	 *
	 * @param[in] pyramid The image pyramid.
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 */
	explicit PyramidPatchExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height);	// Patrik: Why no const here?

	~PyramidPatchExtractor();

	void update(const Mat& image);

	/**
	 * Extracts a patch from the corresponding image pyramid.
	 *
	 * @param[in] x The x-coordinate of the patch center in the original image.
	 * @param[in] y The y-coordinate of the patch center in the original image.
	 * @param[in] width The width of the patch in the original image.
	 * @param[in] height The height of the patch in the original image.
	 * @return The extracted patch or an empty pointer in case the patch could not be extracted.
	 */
	shared_ptr<Patch> extract(int x, int y, int width, int height); // Patrik: Why no const here?

	/**
	 * TODO iterator oder sowas statt vector?!
	 * Extracts several patches from all layers of the corresponding image pyramid.
	 *
	 * @param[in] stepX The step size in x-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] stepY The step size in y-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @return The extracted patches.
	 */
	vector<shared_ptr<Patch>> extract(int stepX, int stepY); // Patrik: Why no const here?

private:

	shared_ptr<ImagePyramid> pyramid; ///< The image pyramid.
	int patchWidth;  ///< The width of the image data of the extracted patches.
	int patchHeight; ///< The height of the image data of the extracted patches.
};

} /* namespace imageprocessing */
#endif /* PYRAMIDPATCHEXTRACTOR_HPP_ */

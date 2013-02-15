/*
 * PatchExtractor.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef PATCHEXTRACTOR_HPP_
#define PATCHEXTRACTOR_HPP_

#include "imageprocessing/Patch.hpp"
#include <memory>
#include <vector>

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace imageprocessing {

class ImagePyramid;
class PyramidLayer;

/**
 * Extracts patches of a constant size from an image pyramid.
 */
class PatchExtractor {
public:

	/**
	 * Constructs a new patch extractor.
	 *
	 * @param[in] pyramid The image pyramid.
	 * @param[in] width The width of the extracted patches.
	 * @param[in] height The height of the extracted patches.
	 */
	explicit PatchExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height);

	~PatchExtractor();

	/**
	 * Updates this patch extractor with a new image.
	 *
	 * @param[in] image The image.
	 */
	void update(const Mat& image);

	/**
	 * Extracts a patch from the corresponding image pyramid.
	 *
	 * @param[in] x The x-coordinate of the patch center in the original image.
	 * @param[in] y The y-coordinate of the patch center in the original image.
	 * @param[in] width The width of the patch in the original image.
	 * @param[in] height The height of the patch in the original image.
	 * @return The extracted patch.
	 */
	unique_ptr<Patch> extract(int x, int y, int width, int height);

	/**
	 * TODO iterator oder sowas statt vector?!
	 * Extracts several patches from all layers of the corresponding image pyramid.
	 *
	 * @param[in] stepX The step size in x-direction in pixels.
	 * @param[in] stepY The step size in y-direction in pixels.
	 * @return The extracted patches.
	 * TODO pointer-zeuch, damit keine kopie?!
	 */
	vector<Patch> extract(int stepX, int stepY);

private:

	/**
	 * Determines the pyramid level that represents the given patch size.
	 *
	 * @param[in] size The patch size.
	 * @return The pyramid level that represents the patch size or null if there is none.
	 */
	PyramidLayer* getPyramidLayer(int size);

private:

	shared_ptr<ImagePyramid> pyramid; ///< The image pyramid.
	int patchWidth;  ///< The width of the extracted patches.
	int patchHeight; ///< The height of the extracted patches.
};

} /* namespace imageprocessing */
#endif /* PATCHEXTRACTOR_HPP_ */

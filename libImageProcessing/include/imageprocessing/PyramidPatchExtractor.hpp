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

using cv::Rect;
using std::vector;

namespace imageprocessing {

class ImagePyramid;
class ImagePyramidLayer;

/**
 * Extracts patches of a constant size from an image pyramid. Does only consider the given width when extracting single
 * patches, as this extractor assumes the given aspect ratio to be the same as the one given at construction, so the
 * extracted patches will not be scaled to fit.
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
	explicit PyramidPatchExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height);

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
	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

	/**
	 * Extracts several patches from the layers of the corresponding image pyramid.
	 * TODO iterator instead of vector?
	 *
	 * @param[in] stepX The step size in x-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] stepY The step size in y-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] roi The region of interest inside the original image (region will be scaled accordingly to the layers).
	 * @param[in] firstLayer The index of the first layer to extract patches from.
	 * @param[in] lastLayer The index of the first layer to extract patches from.
	 * @return The extracted patches.
	 */
	vector<shared_ptr<Patch>> extract(int stepX, int stepY, Rect roi = Rect(), int firstLayer = -1, int lastLayer = -1) const;

	/**
	 * Determines the index of the pyramid layer that approximately contains patches of the given size. Only the width
	 * will be considered.
	 *
	 * @param[in] width The width of the patches.
	 * @param[in] height The height of the patches.
	 * @return The index of the pyramid layer or -1 if there is no layer with an appropriate patch size.
	 */
	int getLayerIndex(int width, int height) const;

private:

	/**
	 * Determines the pyramid layer that approximately contains patches of the given size. Only the width
	 * will be considered.
	 *
	 * @param[in] width The width of the patches.
	 * @param[in] height The height of the patches.
	 * @return The pyramid layer or an empty pointer if there is no layer with an appropriate patch size.
	 */
	const shared_ptr<ImagePyramidLayer> getLayer(int width, int height) const;

	shared_ptr<ImagePyramid> pyramid; ///< The image pyramid.
	int patchWidth;  ///< The width of the image data of the extracted patches.
	int patchHeight; ///< The height of the image data of the extracted patches.
};

} /* namespace imageprocessing */
#endif /* PYRAMIDPATCHEXTRACTOR_HPP_ */

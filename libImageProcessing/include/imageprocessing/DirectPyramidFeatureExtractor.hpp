/*
 * DirectPyramidFeatureExtractor.hpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#ifndef DIRECTPYRAMIDFEATUREEXTRACTOR_HPP_
#define DIRECTPYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/PyramidFeatureExtractor.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include <vector>

using cv::Rect;
using std::vector;

namespace imageprocessing {

class ImageFilter;
class MultipleImageFilter;

/**
 * Pyramid based feature extractor that directly operates on an image pyramid to extract patches of a constant size.
 * Does only consider the given width when extracting single patches, as this extractor assumes the given aspect ratio
 * to be the same as the one given at construction, so the extracted patches will not be scaled to fit.
 */
class DirectPyramidFeatureExtractor : public PyramidFeatureExtractor {
public:

	/**
	 * Constructs a new direct pyramid feature extractor that is based on an image pyramid.
	 *
	 * @param[in] pyramid The image pyramid.
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 */
	DirectPyramidFeatureExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height);

	/**
	 * Constructs a new direct pyramid feature extractor that internally builds its own image pyramid.
	 *
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 * @param[in] minWidth The width of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width of the biggest patches that will be extracted.
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 */
	DirectPyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, double incrementalScaleFactor = 0.85);

	~DirectPyramidFeatureExtractor();

	/**
	 * Adds an image filter to the image pyramid that is applied to the original image.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addImageFilter(shared_ptr<ImageFilter> filter);

	/**
	 * Adds an image filter to the image pyramid that is applied to the down-scaled images.
	 *
	 * @param[in] filter The new layer filter.
	 */
	void addLayerFilter(shared_ptr<ImageFilter> filter);

	/**
	 * Adds a new filter that is applied to the patches.
	 *
	 * @param[in] filter The new patch filter.
	 */
	void addPatchFilter(shared_ptr<ImageFilter> filter);

	void update(const Mat& image) {
		pyramid->update(image);
	}

	void update(shared_ptr<VersionedImage> image) {
		pyramid->update(image);
	}

	/**
	 * Extracts a patch from the corresponding image pyramid. The given width will be used to determine the appropriate
	 * pyramid layer and the patch will be extracted according to the width and height given at construction time. The
	 * patch will not be extracted according to the given height and therefore no rescaling of the height is necessary.
	 * Therefore, the given height is just ignored.
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
	 *
	 * @param[in] stepX The step size in x-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] stepY The step size in y-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] roi The region of interest inside the original image (region will be scaled accordingly to the layers).
	 * @param[in] firstLayer The index of the first layer to extract patches from.
	 * @param[in] lastLayer The index of the last layer to extract patches from.
	 * @return The extracted patches.
	 */
	vector<shared_ptr<Patch>> extract(int stepX, int stepY, Rect roi = Rect(), int firstLayer = -1, int lastLayer = -1) const;

	/**
	 * Extracts a single patch from a layer of the corresponding image pyramid.
	 *
	 * @param[in] layer The index of the layer.
	 * @param[in] x The x-coordinate of the patch center inside the layer.
	 * @param[in] y The y-coordinate of the patch center inside the layer.
	 * @return The extracted patch or an empty pointer in case the patch could not be extracted.
	 */
	shared_ptr<Patch> extract(int layer, int x, int y) const;

	/**
	 * Given a region of interest around patches (patches are completely within the region), compute a region of
	 * interest of the center points of the patches. The given region will be shrunk by half of the patch width
	 * or height on all four sides.
	 *
	 * @param[in] The region of interest of patches that are completely within that region.
	 * @return The region of interest of the same patches, so (at least) their center points are within that region.
	 */
	Rect getCenterRoi(const Rect& roi) const {
		return Rect(roi.x + patchWidth / 2, roi.y + patchHeight / 2, roi.width - patchWidth, roi.height - patchHeight);
	}

	/**
	 * Determines the index of the pyramid layer that approximately contains patches of the given width. The height will
	 * be ignored.
	 *
	 * @param[in] width The width of the patches in the original image.
	 * @param[in] height The height of the patches in the original image.
	 * @return The index of the pyramid layer or -1 if there is no layer with an appropriate patch size.
	 */
	int getLayerIndex(int width, int height) const {
		const shared_ptr<ImagePyramidLayer> layer = getLayer(width);
		return layer ? layer->getIndex() : -1;
	}

	double getMinScaleFactor() const {
		return pyramid->getMinScaleFactor();
	}

	double getMaxScaleFactor() const {
		return pyramid->getMaxScaleFactor();
	}

	double getIncrementalScaleFactor() const {
		return pyramid->getIncrementalScaleFactor();
	}

	Size getImageSize() const {
		return pyramid->getImageSize();
	}

	vector<Size> getLayerSizes() const {
		return pyramid->getLayerSizes();
	}

	vector<Size> getPatchSizes() const;

	/**
	 * @return The image pyramid.
	 */
	shared_ptr<ImagePyramid> getPyramid() {
		return pyramid;
	}

	/**
	 * @return The image pyramid.
	 */
	const shared_ptr<ImagePyramid> getPyramid() const {
		return pyramid;
	}

	/**
	 * @return The width of the image data of the extracted patches.
	 */
	int getPatchWidth() const {
		return patchWidth;
	}

	/**
	 * @return The height of the image data of the extracted patches.
	 */
	int getPatchHeight() const {
		return patchHeight;
	}

private:

	/**
	 * Determines the pyramid layer that approximately contains patches of the given width.
	 *
	 * @param[in] width The width of the patches in the original image.
	 * @return The pyramid layer or an empty pointer if there is no layer with an appropriate patch width.
	 */
	const shared_ptr<ImagePyramidLayer> getLayer(int width) const {
		double scaleFactor = static_cast<double>(patchWidth) / static_cast<double>(width);
		return pyramid->getLayer(scaleFactor);
	}

	shared_ptr<ImagePyramid> pyramid; ///< The image pyramid.
	const int patchWidth;  ///< The width of the image data of the extracted patches.
	const int patchHeight; ///< The height of the image data of the extracted patches.
	shared_ptr<MultipleImageFilter> patchFilter; ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* DIRECTPYRAMIDFEATUREEXTRACTOR_HPP_ */

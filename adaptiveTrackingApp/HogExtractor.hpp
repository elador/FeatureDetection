/*
 * DirectPyramidFeatureExtractor.hpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#ifndef HOGEXTRACTOR_HPP_
#define HOGEXTRACTOR_HPP_

#include "imageprocessing/HistogramFeatureExtractor.hpp"
#include "IntegralHistogramPyramid.hpp"
#include "IntegralHistogramPyramidLayer.hpp"
#include <vector>

using cv::Rect;
using std::vector;

namespace imageprocessing {

/**
 * Pyramid based feature extractor that directly operates on an image pyramid to extract patches of a constant size.
 * Does only consider the given width when extracting single patches, as this extractor assumes the given aspect ratio
 * to be the same as the one given at construction, so the extracted patches will not be scaled to fit.
 */
class HogExtractor : public HistogramFeatureExtractor {
public:

	/**
	 * Constructs a new direct pyramid feature extractor that is based on an image pyramid.
	 *
	 * @param[in] pyramid The image pyramid.
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 */
	HogExtractor(shared_ptr<IntegralHistogramPyramid> pyramid, int width, int height,
			int cellSize, int blockSize, bool combineHistograms = true, Normalization normalization = NONE);

	/**
	 * Constructs a new direct pyramid feature extractor that internally builds its own image pyramid.
	 *
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 * @param[in] minWidth The width of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width of the biggest patches that will be extracted.
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 */
	HogExtractor(int width, int height, int minWidth, int maxWidth, double incrementalScaleFactor,
			int cellSize, int blockSize, bool combineHistograms = true, Normalization normalization = NONE);

	~HogExtractor();

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
		const shared_ptr<IntegralHistogramPyramidLayer> layer = getLayer(width);
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
	shared_ptr<IntegralHistogramPyramid> getPyramid() {
		return pyramid;
	}

	/**
	 * @return The image pyramid.
	 */
	const shared_ptr<IntegralHistogramPyramid> getPyramid() const {
		return pyramid;
	}

	/**
	 * @return The width of the image data of the extracted patches.
	 */
	int getPatchWidth() const {
		return patchWidth;
	}

	/**
	 * @param[in] width The new width of the image data of the extracted patches.
	 */
	void setPatchWidth(int width) {
		patchWidth = width;
	}

	/**
	 * @return The height of the image data of the extracted patches.
	 */
	int getPatchHeight() const {
		return patchHeight;
	}

	/**
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchHeight(int height) {
		patchHeight = height;
	}

	/**
	 * Changes the size of the image data of the extracted patches.
	 *
	 * @param[in] width The new width of the image data of the extracted patches.
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchSize(int width, int height) {
		patchWidth = width;
		patchHeight = height;
	}

	mutable Mat image;

private:

	/**
	 * Determines the pyramid layer that approximately contains patches of the given width.
	 *
	 * @param[in] width The width of the patches in the original image.
	 * @return The pyramid layer or an empty pointer if there is no layer with an appropriate patch width.
	 */
	const shared_ptr<IntegralHistogramPyramidLayer> getLayer(int width) const {
		double scaleFactor = static_cast<double>(patchWidth) / static_cast<double>(width);
		return pyramid->getLayer(scaleFactor);
	}

	Mat getHogFeature(const vector<Mat>& integralHistogram) const;

	shared_ptr<IntegralHistogramPyramid> pyramid; ///< The image pyramid.
	int patchWidth;  ///< The width of the image data of the extracted patches.
	int patchHeight; ///< The height of the image data of the extracted patches.

	int cellWidth;     ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;    ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;    ///< The width of the blocks in cells.
	int blockHeight;   ///< The height of the blocks in cells.
	bool combineHistograms; ///< Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
};

} /* namespace imageprocessing */
#endif /* HOGEXTRACTOR_HPP_ */

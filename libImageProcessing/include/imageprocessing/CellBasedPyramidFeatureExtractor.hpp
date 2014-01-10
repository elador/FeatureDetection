/*
 * CellBasedPyramidFeatureExtractor.hpp
 *
 *  Created on: 16.12.2013
 *      Author: poschmann
 */

#ifndef CELLBASEDPYRAMIDFEATUREEXTRACTOR_HPP_
#define CELLBASEDPYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/PyramidFeatureExtractor.hpp"

namespace imageprocessing {

class ImagePyramid;
class ImagePyramidLayer;
class ImageFilter;
class ChainedFilter;

/**
 * Pyramid feature extractor whose patches consist of rectangular cells that contain several pixels. Each cell
 * is described by a descriptor.
 */
class CellBasedPyramidFeatureExtractor : public PyramidFeatureExtractor {
public:

	/**
	 * Constructs a new cell based pyramid feature extractor.
	 *
	 * @param[in] cellSize The size (width and height) of the cells in pixels.
	 * @param[in] cols The cell column count of the extracted patches.
	 * @param[in] rows The cell row count of the extracted patches.
	 * @param[in] minWidth The width (in pixels) of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width (in pixels) of the biggest patches that will be extracted.
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 */
	CellBasedPyramidFeatureExtractor(size_t cellSize, size_t cols, size_t rows,
			size_t minWidth, size_t maxWidth, double incrementalScaleFactor = 0.85);

	/**
	 * Adds an image filter to the image pyramid that is applied to the original image.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addImageFilter(shared_ptr<ImageFilter> filter);

	/**
	 * Adds an image filter to the image pyramid that is applied to the down-scaled images. One of the
	 * layer filters should be responsible for reducing the image to cells of the size that was specified
	 * at the creation of this extractor.
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

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

	vector<shared_ptr<Patch>> extract(int stepX, int stepY, Rect roi = Rect(),
			int firstLayer = -1, int lastLayer = -1, int stepLayer = 1) const;

	shared_ptr<Patch> extract(int layer, int x, int y) const;

	Rect getCenterRoi(const Rect& roi) const;

	int getLayerIndex(int width, int height) const;

	double getMinScaleFactor() const;

	double getMaxScaleFactor() const;

	double getIncrementalScaleFactor() const;

	Size getPatchSize() const;

	Size getImageSize() const;

	vector<pair<int, double>> getLayerScales() const;

	vector<Size> getLayerSizes() const;

	vector<Size> getPatchSizes() const;

private:

	/**
	 * Determines the pyramid layer that approximately contains patches of the given width.
	 *
	 * @param[in] width The width of the patches in the original image.
	 * @return The pyramid layer or an empty pointer if there is no layer with an appropriate patch width.
	 */
	const shared_ptr<ImagePyramidLayer> getLayer(size_t width) const;

	shared_ptr<ImagePyramid> pyramid; ///< The image pyramid.
	size_t cellSize;        ///< The size (width and height) of the cells in pixels.
	size_t cellColumnCount; ///< The cell column count of the extracted patches.
	size_t cellRowCount;    ///< The cell row count of the extracted patches.
	shared_ptr<ChainedFilter> patchFilter; ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* CELLBASEDPYRAMIDFEATUREEXTRACTOR_HPP_ */

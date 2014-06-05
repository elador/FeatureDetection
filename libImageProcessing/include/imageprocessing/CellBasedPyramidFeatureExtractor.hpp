/*
 * CellBasedPyramidFeatureExtractor.hpp
 *
 *  Created on: 16.12.2013
 *      Author: poschmann
 */

#ifndef CELLBASEDPYRAMIDFEATUREEXTRACTOR_HPP_
#define CELLBASEDPYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"

namespace imageprocessing {

class ImagePyramid;
class ImagePyramidLayer;
class ImageFilter;
class ChainedFilter;

/**
 * Pyramid feature extractor whose patches consist of square cells that contain several pixels. Each cell
 * is described by a descriptor.
 */
class CellBasedPyramidFeatureExtractor : public DirectPyramidFeatureExtractor {
public:

	/**
	 * Constructs a new cell based pyramid feature extractor that internally builds its own image pyramid.
	 *
	 * @param[in] cellSize The size (width and height) of the cells in pixels.
	 * @param[in] cols The cell column count of the extracted patches.
	 * @param[in] rows The cell row count of the extracted patches.
	 * @param[in] minWidth The width (in pixels) of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width (in pixels) of the biggest patches that will be extracted.
	 * @param[in] octaveLayerCount The number of layers per octave.
	 */
	CellBasedPyramidFeatureExtractor(int cellSize, int cols, int rows,
			int minWidth, int maxWidth, int octaveLayerCount = 5);

	/**
	 * Constructs a new cell based pyramid feature extractor that is based on a given image pyramid.
	 *
	 * @param[in] pyramid The image pyramid.
	 * @param[in] cellSize The size (width and height) of the cells in pixels.
	 * @param[in] cols The cell column count of the extracted patches.
	 * @param[in] rows The cell row count of the extracted patches.
	 */
	CellBasedPyramidFeatureExtractor(std::shared_ptr<ImagePyramid> pyramid, int cellSize, int cols, int rows);

	std::vector<std::shared_ptr<Patch>> extract(int stepX, int stepY, cv::Rect roi = cv::Rect(),
			int firstLayer = -1, int lastLayer = -1, int stepLayer = 1) const;

	std::vector<std::pair<int, double>> getLayerScales() const;

protected:

	int getScaled(const ImagePyramidLayer& layer, int value) const;

	int getOriginal(const ImagePyramidLayer& layer, int value) const;

	const std::shared_ptr<ImagePyramidLayer> getLayer(int width) const;

private:

	size_t cellSize;        ///< The size (width and height) of the cells in pixels.
	size_t realPatchWidth;  ///< The width of the extracted patch data in pixels.
};

} /* namespace imageprocessing */
#endif /* CELLBASEDPYRAMIDFEATUREEXTRACTOR_HPP_ */

/*
 * SpatialHistogramFilter.hpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#ifndef SPATIALHISTOGRAMFILTER_HPP_
#define SPATIALHISTOGRAMFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"

namespace imageprocessing {

/**
 * Filter that creates spatial histograms that may overlap. The input is an image with depth CV_8U containing bin information
 * (bin indices and optionally weights). It must have one, two or four channels. In case of one channel, it just contains the
 * bin index. The second channel adds a weight that will be divided by 255 to be between zero and one. The third channel is
 * another bin index and the fourth channel is the corresponding weight. The output will be a row vector of type CV_32F
 * containing the concatenated normalized histograms.
 *
 * If the spatial histograms are non-overlapping (each block consists of only one cell), then the normalization will be applied
 * afterwards over the whole vector of concatenated histograms instead of to the individual histograms.
 */
class SpatialHistogramFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new spatial histogram filter with square cells and blocks.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] blockSize The width and height of the blocks in cells.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] concatenate Flag that indicates whether the cell histograms of a block should be concatenated instead of added.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFilter(int binCount, int cellSize, int blockSize,
			bool interpolate, bool concatenate = false, Normalization normalization = Normalization::NONE);

	/**
	 * Constructs a new spatial histogram filter.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] blockWidth The width of the blocks in cells.
	 * @param[in] blockHeight The height of the blocks in cells.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] concatenate Flag that indicates whether the cell histograms of a block should be concatenated instead of added.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFilter(int binCount, int cellWidth, int cellHeight, int blockWidth, int blockHeight,
			bool interpolate, bool concatenate = false, Normalization normalization = Normalization::NONE);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	/**
	 * Creates block histograms that cover several cell histograms.
	 *
	 * @param[in] cellHistograms Row vector containing the histogram values of the cells in row-major order.
	 * @param[out] blockHistograms Row vector containing the histogram values of the blocks in row-major order.
	 * @param[in] binCount Bin count of the histograms.
	 * @param[in] cellRowCount Row count of the cell grid.
	 * @param[in] cellColumnCount Column count of the cell grid.
	 * @param[in] blockWidth Width of the blocks (in amount of cells).
	 * @param[in] blockHeight Height of the blocks (in amount of cells).
	 * @param[in] concatenate Flag that indicates whether the cell histograms of a block should be concatenated instead of added.
	 */
	void createBlockHistograms(const cv::Mat& cellHistograms, cv::Mat& blockHistograms, int binCount,
			int cellRowCount, int cellColumnCount, int blockWidth, int blockHeight, bool concatenate) const;

	int binCount;    ///< The amount of bins inside the histogram.
	int cellWidth;   ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;  ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;  ///< The width of the blocks in cells.
	int blockHeight; ///< The height of the blocks in cells.
	bool interpolate; ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	bool concatenate; ///< Flag that indicates whether the cell histograms of a block should be concatenated instead of added.
};

} /* namespace imageprocessing */
#endif /* SPATIALHISTOGRAMFILTER_HPP_ */

/*
 * HogFilter.hpp
 *
 *  Created on: 26.08.2013
 *      Author: poschmann
 */

#ifndef HOGFILTER_HPP_
#define HOGFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"

namespace imageprocessing {

/**
 * Filter that creates histograms of oriented gradients (HOG) feature vectors based on images containing gradient bin
 * information. The normalization method is L2NORM and cannot be changed.
 *
 * The input is an image with depth CV_8U containing bin information (bin indices and optionally weights). It must have
 * one, two or four channels. In case of one channel, it just contains the bin index. The second channel adds a weight
 * that will be divided by 255 to be between zero and one. The third channel is another bin index and the fourth channel
 * is the corresponding weight. The output will be a row vector of type CV_32F.
 */
class HogFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new spatial HOG filter with square cells and blocks.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] blockSize The width and height of the blocks in cells.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	HogFilter(int binCount, int cellSize, int blockSize, bool interpolate = false, bool signedAndUnsigned = false);

	/**
	 * Constructs a new spatial HOG filter.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] blockWidth The width of the blocks in cells.
	 * @param[in] blockHeight The height of the blocks in cells.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	HogFilter(int binCount, int cellWidth, int cellHeight, int blockWidth, int blockHeight,
			bool interpolate = false, bool signedAndUnsigned = false);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	/**
	 * Creates block histograms that cover several cell histograms.
	 *
	 * @param[in] cellHistograms Row vector containing the histogram values of the cells in row-major order.
	 * @param[in] cellEnergies Row vector containing the gradient energies of the cells in row-major order.
	 * @param[out] blockHistograms Row vector containing the histogram values of the blocks in row-major order.
	 * @param[in] binCount Bin count of the histograms.
	 * @param[in] cellRowCount Row count of the cell grid.
	 * @param[in] cellColumnCount Column count of the cell grid.
	 * @param[in] blockWidth Width of the blocks (in amount of cells).
	 * @param[in] blockHeight Height of the blocks (in amount of cells).
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	void createBlockHistograms(const cv::Mat& cellHistograms, const cv::Mat& cellEnergies, cv::Mat& blockHistograms,
			int binCount, int cellRowCount, int cellColumnCount, int blockWidth, int blockHeight, bool signedAndUnsigned) const;

	/**
	 * Computes the gradient energies of the cells.
	 *
	 * @param[in] cellHistograms Row vector containing the histogram values of the cells in row-major order.
	 * @param[out] cellEnergies Row vector containing the gradient energies of the cells in row-major order.
	 * @param[in] binCount Bin count of the histograms.
	 * @param[in] cellRowCount Row count of the cell grid.
	 * @param[in] cellColumnCount Column count of the cell grid.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	void computeCellEnergies(const cv::Mat& cellHistograms, cv::Mat& cellEnergies,
			int binCount, int cellRowCount, int cellColumnCount, bool signedAndUnsigned) const;

	int binCount;    ///< The amount of bins inside the histogram.
	int cellWidth;   ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;  ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;  ///< The width of the blocks in cells.
	int blockHeight; ///< The height of the blocks in cells.
	bool interpolate;       ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	bool signedAndUnsigned; ///< Flag that indicates whether signed and unsigned gradients should be used.
};

} /* namespace imageprocessing */
#endif /* HOGFILTER_HPP_ */

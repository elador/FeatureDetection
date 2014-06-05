/*
 * ExtendedHogFilter.hpp
 *
 *  Created on: 01.08.2013
 *      Author: poschmann
 */

#ifndef EXTENDEDHOGFILTER_HPP_
#define EXTENDEDHOGFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"
#include <vector>

namespace imageprocessing {

/**
 * Filter that builds extended HOG feature vectors. The extended HOG features were described in [1]. But instead of
 * computing the HOG descriptors for the inner cells only, we compute the descriptors for the border cells, too. The
 * necessary gradient energies, that would have computed from cells outside of the image, are taken from the nearest
 * cells within the image - so the gradient energies of the border cells are repeated for the imaginary cells right
 * outside the image.
 *
 * The input is an image with depth CV_8U containing bin information (bin indices and optionally weights). It must have
 * one, two or four channels. In case of one channel, it just contains the bin index. The second channel adds a weight
 * that will be divided by 255 to be between zero and one. The third channel is another bin index and the fourth channel
 * is the corresponding weight. The output will be an image containing the HOG cell descriptors. Its depth is CV_32F and
 * the amount of channels depends on the amount of values of each cell descriptor. If both signed and unsigned gradients
 * should be integrated into the descriptor, the binCount describes the amount of signed gradient bins. In that case,
 * the descriptor will have a size of binCount + binCount / 2 + 4, otherwise it is binCount + 4.
 *
 * [1] Felzenszwalb et al., Object Detection with Discriminatively Trained Part-Based Models, PAMI, 2010.
 */
class ExtendedHogFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new extended HOG filter with square cells and blocks.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	ExtendedHogFilter(int binCount, int cellSize, bool interpolate, bool signedAndUnsigned, float alpha = 0.2);

	/**
	 * Constructs a new extended HOG filter.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	ExtendedHogFilter(int binCount, int cellWidth, int cellHeight, bool interpolate, bool signedAndUnsigned, float alpha = 0.2);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	/**
	 * @return The width of the cells.
	 */
	int getCellWidth() {
		return cellWidth;
	}

	/**
	 * @return The height of the cells.
	 */
	int getCellHeight() {
		return cellHeight;
	}

private:

	/**
	 * Creates cell descriptors based on a grid of cell histograms.
	 *
	 * @param[in] histograms Row vector containing the histogram values of the cells in row-major order.
	 * @param[out] descriptors Row vector containing the descriptors of the cells in row-major order.
	 * @param[in] bins Bin count of the histograms.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] cellRows Row count of the cell grid.
	 * @param[in] cellCols Column count of the cell grid.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	void createDescriptors(const cv::Mat& histograms, cv::Mat& descriptors,
			int bins, bool signedAndUnsigned, int cellRows, int cellCols, float alpha) const;

	int binCount;   ///< The amount of bins inside the histogram.
	int cellWidth;  ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight; ///< The preferred height of the cells in pixels (actual height might deviate).
	bool interpolate;       ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	bool signedAndUnsigned; ///< Flag that indicates whether signed and unsigned gradients should be used.
	float alpha; ///< Truncation threshold of the orientation bin values (applied after normalization).
};

} /* namespace imageprocessing */
#endif /* EXTENDEDHOGFILTER_HPP_ */

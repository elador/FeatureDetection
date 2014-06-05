/*
 * CompleteExtendedHogFilter.hpp
 *
 *  Created on: 06.01.2014
 *      Author: poschmann
 */

#ifndef COMPLETEEXTENDEDHOGFILTER_HPP_
#define COMPLETEEXTENDEDHOGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <vector>
#include <array>

namespace imageprocessing {

/**
 * Image filter that takes a grayscale image and computes extended HOG descriptors for cells of a certain size. The
 * output image will have one descriptor per row/column pair and have as many channels as there are values inside
 * the descriptors.
 *
 * The extended HOG features were described in [1]. But instead of computing the HOG descriptors for the inner cells
 * only, we compute the descriptors for the border cells, too. The necessary gradient energies, that would have
 * computed from cells outside of the image, are taken from the nearest cells within the image - so the gradient
 * energies of the border cells are repeated for the imaginary cells right outside the image.
 *
 * The input image has to be of type CV_8UC1, the output image will be of type CV_32FC(#), where # depends on the
 * number of orientation bins and on whether signed and unsigned gradients should be combined. If both signed and
 * unsigned gradients should be integrated into the descriptor, the binCount describes the amount of signed gradient
 * bins. In that case, the descriptor will have a size of binCount + binCount / 2 + 4, otherwise it is
 * binCount + 4.
 *
 * [1] Felzenszwalb et al., Object Detection with Discriminatively Trained Part-Based Models, PAMI, 2010.
 */
class CompleteExtendedHogFilter : public ImageFilter {
public:

	/**
	 * Constructs a new complete extended HOG filter.
	 *
	 * @param[in] cellSize The width and height of the cells.
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] signedGradients Flag that indicates whether signed gradients (360°) should be computed.
	 * @param[in] unsignedGradients Flag that indicates whether unsigned gradients (180°) should be computed.
	 * @param[in] interpolateBins Flag that indicates whether a gradient should contribute to two neighboring bins in a weighted manner.
	 * @param[in] interpolateCells Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	explicit CompleteExtendedHogFilter(size_t cellSize = 8, size_t binCount = 18, bool signedGradients = true, bool unsignedGradients = true,
			bool interpolateBins = false, bool interpolateCells = true, float alpha = 0.2f);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	/**
	 * @return The width and height of the cells.
	 */
	size_t getCellSize() {
		return cellSize;
	}

private:

	/**
	 * Information about values being added to histogram bins.
	 */
	struct BinInformation {
		int index1; ///< Index of the first bin.
		int index2; ///< Index of the second bin.
		float weight1; ///< Weight of the first bin.
		float weight2; ///< Weight of the second bin.
	};

	/**
	 * Creates the look-up-table if its size does not match the requested size.
	 *
	 * @param[in,out] lut The look-up table.
	 * @param[in] size The necessary amount of entries (row/column count).
	 * @param[in] count The number of cells.
	 */
	void createLut(std::vector<BinInformation>& lut, size_t size, size_t count) const;

	/**
	 * Creates the initial histograms by computing the gradients over the image.
	 *
	 * @param[in,out] histograms The (empty) histogram/descriptor of each cell.
	 * @param[in] image The image.
	 * @param[in] cellRowCount The cell row count that fits into the image.
	 * @param[in] cellColumnCount The cell column count that fits into the image.
	 */
	void buildInitialHistograms(cv::Mat& histograms, const cv::Mat& image, size_t cellRowCount, size_t cellColumnCount) const;

	/**
	 * Builds the descriptor of each cell using its histogram data.
	 *
	 * @param[in,out] descriptors The descriptors (initially containing the histogram data) of each cell.
	 * @param[in] cellRowCount The cell row count that fits into the image.
	 * @param[in] cellColumnCount The cell column count that fits into the image.
	 * @param[in] descriptorSize The value count of the descriptors.
	 */
	void buildDescriptors(cv::Mat& descriptors, size_t cellRowCount, size_t cellColumnCount, size_t descriptorSize) const;

	size_t cellSize; ///< The width and height of the cells.
	size_t binCount; ///< The amount of bins inside the histogram.
	bool signedGradients;   ///< Flag that indicates whether signed gradients (360°) should be computed.
	bool unsignedGradients; ///< Flag that indicates whether unsigned gradients (180°) should be computed.
	bool interpolateBins;   ///< Flag that indicates whether a gradient should contribute to two neighboring bins in a weighted manner.
	bool interpolateCells;  ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	float alpha; ///< Truncation threshold of the orientation bin values (applied after normalization).

	std::array<BinInformation, 512 * 512> binLut;  ///< Look-up table for bin information given a gradient code.
	mutable std::vector<BinInformation> rowLut;    ///< Look-up table for the linear interpolation of the row indices.
	mutable std::vector<BinInformation> columnLut; ///< Look-up table for the linear interpolation of the column indices.

	static const float eps; ///< The small value being added to the norm to prevent division by zero.
};

} /* namespace imageprocessing */
#endif /* COMPLETEEXTENDEDHOGFILTER_HPP_ */

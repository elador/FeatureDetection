/*
 * VlHogFilter.cpp
 *
 *  Created on: 03.07.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef VLHOGFILTER_HPP_
#define VLHOGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

#include <string>

extern "C" {
	#include "superviseddescent/hog.h"
}

namespace superviseddescent { // We'll move it to imageprocessing once finished

/**
 * Filter that ...
 * Returns a Matrix, as many rows as points, 1 descriptor = 1 row
 */
class VlHogFilter : public imageprocessing::ImageFilter {
public:
	
	enum class VlHogType {
		DalalTriggs,
		Uoctti
	};

	/**
	 * Constructs a new extended HOG filter with square cells and blocks.
	 *
	 * @param[in] vlhogType The amount of bins inside the histogram.
	 * @param[in] numCells The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] cellSize Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] numBins Flag that indicates whether signed and unsigned gradients should be used.
	 */
	VlHogFilter(VlHogType vlhogType, int numCells, int cellSize, int numBins);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:
	VlHogType hogType;
	int numCells;
	int cellSize;
	int numBins;

	friend std::string getParameterString(VlHogFilter filter); // Hmm let's see if we find a better solution for this, none of the other ImageFilters has it. Kind of belongs in the app/config reading, not here. Provide getters here.
};

std::string getParameterString(VlHogFilter filter) {
	return std::string("numCells " + std::to_string(filter.numCells) + " cellSize " + std::to_string(filter.cellSize) + " numBins " + std::to_string(filter.numBins));
};

} /* namespace superviseddescent */
#endif /* VLHOGFILTER_HPP_ */

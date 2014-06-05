/*
 * IntegralChannelHistogramFilter.hpp
 *
 *  Created on: 06.08.2013
 *      Author: poschmann
 */

#ifndef INTEGRALCHANNELHISTOGRAMFILTER_HPP_
#define INTEGRALCHANNELHISTOGRAMFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"

namespace imageprocessing {

/**
 * Filter that expects each channel of the given images to be an integral histogram bin. The filter result is a
 * vector of concatenated normalized histograms over overlapping blocks. The input images must have a depth of
 * CV_32S, the output vectors will have a depth of CV_32F.
 */
class IntegralChannelHistogramFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new integral channel histogram filter.
	 *
	 * @param[in] blockRows The amount of rows of blocks.
	 * @param[in] blockColumns The amount of columns of blocks.
	 * @param[in] overlap The overlap between neighboring blocks.
	 * @param[in] normalization The normalization method of the histograms.
	 */
	IntegralChannelHistogramFilter(unsigned int blockRows, unsigned int blockColumns, double overlap = 0,
			Normalization normalization = Normalization::L2NORM);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	unsigned int blockRows;    ///< The amount of rows of blocks.
	unsigned int blockColumns; ///< The amount of columns of blocks.
	double overlap; ///< The overlap between neighboring blocks.
};

} /* namespace imageprocessing */
#endif /* INTEGRALCHANNELHISTOGRAMFILTER_HPP_ */

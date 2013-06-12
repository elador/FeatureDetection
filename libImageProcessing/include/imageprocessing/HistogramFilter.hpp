/*
 * HistogramFilter.hpp
 *
 *  Created on: 12.06.2013
 *      Author: poschmann
 */

#ifndef HISTOGRAMFILTER_HPP_
#define HISTOGRAMFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that produces and image containing bin information of each pixel.
 */
class HistogramFilter : public ImageFilter {
public:

	virtual ~HistogramFilter() {}

	/**
	 * @return The amount of bins.
	 */
	virtual unsigned int getBinCount() const = 0;
};

} /* namespace imageprocessing */
#endif /* HISTOGRAMFILTER_HPP_ */

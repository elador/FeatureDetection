/*
 * BinningFilter.hpp
 *
 *  Created on: 12.06.2013
 *      Author: poschmann
 */

#ifndef BINNINGFILTER_HPP_
#define BINNINGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter whose result is an image containing bin information of each pixel.
 */
class BinningFilter : public ImageFilter {
public:

	virtual ~BinningFilter() {}

	/**
	 * @return The amount of bins.
	 */
	virtual unsigned int getBinCount() const = 0;
};

} /* namespace imageprocessing */
#endif /* BINNINGFILTER_HPP_ */

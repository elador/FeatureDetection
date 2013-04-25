/*
 * ReshapingFilter.hpp
 *
 *  Created on: 25.04.2013
 *      Author: poschmann
 */

#ifndef RESHAPINGFILTER_HPP_
#define RESHAPINGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Filter that reshapes images.
 */
class ReshapingFilter : public ImageFilter {
public:

	/**
	 * Constructs a new reshaping filter.
	 *
	 * @param[in] rows The new number of rows. If zero, the number of rows remains the same.
	 * @param[in] channels The new number of channels. If zero, the number of channels remains the same.
	 */
	explicit ReshapingFilter(int rows, int channels = 0);

	~ReshapingFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);

private:

	int rows;     ///< The new number of rows. If zero, the number of rows remains the same.
	int channels; ///< The new number of channels. If zero, the number of channels remains the same.
};

} /* namespace imageprocessing */
#endif /* RESHAPINGFILTER_HPP_ */

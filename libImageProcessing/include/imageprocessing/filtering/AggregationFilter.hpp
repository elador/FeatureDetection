/*
 * AggregationFilter.hpp
 *
 *  Created on: 14.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_AGGREGATIONFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_AGGREGATIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <memory>

namespace imageprocessing {
namespace filtering {

/**
 * Filter that aggregates pixel values over square cells.
 *
 * Places a filter kernel over the center pixel of the cells. The response of the filter are the aggregated
 * values over the cell. If each pixel should only contribute to the cell it is contained in (no interpolation),
 * then a box filter of the size of the cell is used. Otherwise, a triangular filter twice as big as a cell is
 * used, so each pixel effectively contributes to (at most) four cells using bilinear interpolation.
 *
 * Without normalization, the aggregated values are the sum of the pixel values that contribute to the cell.
 * With normalization, this value is divided by the area of the cell, yielding the mean value over the cell.
 */
class AggregationFilter : public ImageFilter {
public:

	/**
	 * Constructs a new aggregation filter.
	 *
	 * @param[in] cellSize Size of the square cells in pixels.
	 * @param[in] interpolate Flag that indicates whether to bilinearly interpolate the pixel contributions.
	 * @param[in] normalize Flag that indicates whether the sum should be normalized by the area, yielding the mean.
	 */
	explicit AggregationFilter(int cellSize, bool interpolate = false, bool normalize = false);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	std::unique_ptr<ImageFilter> downsamplingConvolutionFilter; ///< Actual filter used for the aggregation.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_AGGREGATIONFILTER_HPP_ */

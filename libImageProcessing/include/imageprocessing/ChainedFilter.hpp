/*
 * ChainedFilter.hpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#ifndef CHAINEDFILTER_HPP_
#define CHAINEDFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <vector>
#include <memory>

namespace imageprocessing {

/**
 * Image filter that applies several filters in order. The result of a filter is used as input of the following filter,
 * the result of the last filter is the result of the whole filter chain.
 */
class ChainedFilter : public ImageFilter {
public:

	/**
	 * Constructs a new empty chained image filter.
	 */
	ChainedFilter();

	/**
	 * Constructs a new chained image filter.
	 *
	 * @param[in] filters The filters in order of application.
	 */
	explicit ChainedFilter(std::vector<std::shared_ptr<ImageFilter>> filters);

	/**
	 * Constructs a new chained image filter with two filters.
	 *
	 * @param[in] filter1 The first image filter.
	 * @param[in] filter2 The second image filter.
	 */
	ChainedFilter(std::shared_ptr<ImageFilter> filter1, std::shared_ptr<ImageFilter> filter2);

	/**
	 * Constructs a new chained image filter with three filters.
	 *
	 * @param[in] filter1 The first image filter.
	 * @param[in] filter2 The second image filter.
	 * @param[in] filter3 The third image filter.
	 */
	ChainedFilter(std::shared_ptr<ImageFilter> filter1, std::shared_ptr<ImageFilter> filter2, std::shared_ptr<ImageFilter> filter3);

	/**
	 * Adds a new image filter that is applied after the currently existing filters.
	 *
	 * @param[in] filter The new image filter.
	 */
	void add(std::shared_ptr<ImageFilter> filter);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	void applyInPlace(cv::Mat& image) const;

private:

	std::vector<std::shared_ptr<ImageFilter>> filters; ///< The filters in order of application.
};

} /* namespace imageprocessing */
#endif /* CHAINEDFILTER_HPP_ */

/*
 * MultipleImageFilter.hpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#ifndef MULTIPLEIMAGEFILTER_HPP_
#define MULTIPLEIMAGEFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

namespace imageprocessing {

/**
 * Image filter that applies several filters in order.
 */
class MultipleImageFilter : public ImageFilter {
public:

	/**
	 * Constructs a new empty multiple image filter.
	 */
	MultipleImageFilter();

	~MultipleImageFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

	void applyInPlace(Mat& image) const;

	/**
	 * Adds a new image filter that is applied to images after the currently existing filters.
	 *
	 * @param[in] filter The new image filter.
	 */
	void add(shared_ptr<ImageFilter> filter) {
		filters.push_back(filter);
	}

private:

	vector<shared_ptr<ImageFilter>> filters; ///< The filters in order.
};

} /* namespace imageprocessing */
#endif /* MULTIPLEIMAGEFILTER_HPP_ */

/*
 * ChainedFilter.cpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/ChainedFilter.hpp"

using cv::Mat;
using std::vector;
using std::shared_ptr;

namespace imageprocessing {

ChainedFilter::ChainedFilter() : filters() {}

ChainedFilter::ChainedFilter(vector<shared_ptr<ImageFilter>> filters) : filters(filters) {}

ChainedFilter::ChainedFilter(shared_ptr<ImageFilter> filter1, shared_ptr<ImageFilter> filter2) : filters(2) {
	filters[0] = filter1;
	filters[1] = filter2;
}

ChainedFilter::ChainedFilter(
		shared_ptr<ImageFilter> filter1, shared_ptr<ImageFilter> filter2, shared_ptr<ImageFilter> filter3) : filters(3) {
	filters[0] = filter1;
	filters[1] = filter2;
	filters[2] = filter3;
}

void ChainedFilter::add(shared_ptr<ImageFilter> filter) {
	filters.push_back(filter);
}

Mat ChainedFilter::applyTo(const Mat &image, Mat &filtered) const {
	if (filters.empty()) {
		image.copyTo(filtered);
	} else {
		filters[0]->applyTo(image, filtered);
		for (unsigned int i = 1; i < filters.size(); ++i)
			filters[i]->applyInPlace(filtered);
	}
	return filtered;
}

void ChainedFilter::applyInPlace(Mat& image) const {
	for (unsigned int i = 0; i < filters.size(); ++i)
		filters[i]->applyInPlace(image);
}

} /* namespace imageprocessing */

/*
 * ParallelFilter.cpp
 *
 *  Created on: 08.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/ParallelFilter.hpp"

using cv::Mat;
using std::vector;
using std::shared_ptr;

namespace imageprocessing {

ParallelFilter::ParallelFilter() : filters() {}

ParallelFilter::ParallelFilter(vector<shared_ptr<ImageFilter>> filters) : filters(filters) {}

ParallelFilter::ParallelFilter(shared_ptr<ImageFilter> filter1, shared_ptr<ImageFilter> filter2) : filters(2) {
	filters[0] = filter1;
	filters[1] = filter2;
}

ParallelFilter::ParallelFilter(
		shared_ptr<ImageFilter> filter1, shared_ptr<ImageFilter> filter2, shared_ptr<ImageFilter> filter3) : filters(3) {
	filters[0] = filter1;
	filters[1] = filter2;
	filters[2] = filter3;
}

void ParallelFilter::add(shared_ptr<ImageFilter> filter) {
	filters.push_back(filter);
}

Mat ParallelFilter::applyTo(const Mat& image, Mat& filtered) const {
	vector<Mat> results(filters.size());
	for (unsigned int i = 0; i < filters.size(); ++i)
		filters[i]->applyTo(image, results[i]);
	cv::merge(results, filtered);
	return filtered;
}

} /* namespace imageprocessing */

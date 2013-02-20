/*
 * MultipleImageFilter.cpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/MultipleImageFilter.hpp"

namespace imageprocessing {

MultipleImageFilter::MultipleImageFilter() : filters() {}

MultipleImageFilter::~MultipleImageFilter() {}

Mat MultipleImageFilter::applyTo(const Mat &image, Mat &filtered) {
	if (filters.empty()) {
		image.copyTo(filtered);
	} else {
		filters[0]->applyTo(image, filtered);
		for (unsigned int i = 1; i < filters.size(); ++i)
			filters[i]->applyInPlace(filtered);
	}
	return filtered;
}

void MultipleImageFilter::applyInPlace(Mat& image) {
	for (unsigned int i = 0; i < filters.size(); ++i)
		filters[i]->applyInPlace(image);
}

} /* namespace imageprocessing */

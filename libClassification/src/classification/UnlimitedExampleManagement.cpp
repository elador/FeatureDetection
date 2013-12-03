/*
 * UnlimitedExampleManagement.cpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#include "classification/UnlimitedExampleManagement.hpp"

using cv::Mat;
using std::vector;

namespace classification {

UnlimitedExampleManagement::UnlimitedExampleManagement(size_t requiredSize) : VectorBasedExampleManagement(10, requiredSize) {}

void UnlimitedExampleManagement::add(const vector<Mat>& newExamples) {
	for (const Mat& example : newExamples)
		examples.push_back(example);
}

} /* namespace classification */

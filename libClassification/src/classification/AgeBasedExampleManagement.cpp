/*
 * AgeBasedExampleManagement.cpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#include "classification/AgeBasedExampleManagement.hpp"

using cv::Mat;
using std::vector;

namespace classification {

AgeBasedExampleManagement::AgeBasedExampleManagement(size_t capacity, size_t requiredSize) :
		VectorBasedExampleManagement(capacity, requiredSize), insertPosition(0) {}

void AgeBasedExampleManagement::add(const vector<Mat>& newExamples) {
	// add new training examples as long as there is space available
	auto example = newExamples.cbegin();
	for (; examples.size() < examples.capacity() && example != newExamples.cend(); ++example)
		examples.push_back(*example);
	// replace the oldest training examples by new ones
	for (; example != newExamples.cend(); ++example) {
		examples[insertPosition] = *example;
		++insertPosition;
		if (insertPosition == examples.size())
			insertPosition = 0;
	}
}

} /* namespace classification */

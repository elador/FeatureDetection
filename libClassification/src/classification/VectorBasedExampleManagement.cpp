/*
 * VectorBasedExampleManagement.cpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#include "classification/VectorBasedExampleManagement.hpp"

using cv::Mat;
using std::vector;
using std::unique_ptr;

namespace classification {

VectorBasedExampleManagement::VectorBasedExampleManagement(size_t capacity, size_t requiredSize) :
		examples(), requiredSize(requiredSize) {
	examples.reserve(capacity);
}

VectorBasedExampleManagement::~VectorBasedExampleManagement() {}

void VectorBasedExampleManagement::clear() {
	examples.clear();
}

size_t VectorBasedExampleManagement::size() const {
	return examples.size();
}

bool VectorBasedExampleManagement::hasRequiredSize() const {
	return examples.size() >= requiredSize;
}

unique_ptr<ExampleManagement::ExampleIterator> VectorBasedExampleManagement::iterator() const {
	return unique_ptr<VectorIterator>(new VectorIterator(examples));
}

VectorBasedExampleManagement::VectorIterator::VectorIterator(const vector<Mat>& examples) : current(examples.cbegin()), end(examples.cend()) {}

bool VectorBasedExampleManagement::VectorIterator::hasNext() const {
	return current != end;
}

const Mat& VectorBasedExampleManagement::VectorIterator::next() {
	const Mat& example = *current;
	++current;
	return example;
}

} /* namespace classification */

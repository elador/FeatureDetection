/*
 * FrameBasedExampleManagement.cpp
 *
 *  Created on: 28.11.2013
 *      Author: poschmann
 */

#include "classification/FrameBasedExampleManagement.hpp"

using cv::Mat;
using std::vector;
using std::unique_ptr;

namespace classification {

FrameBasedExampleManagement::FrameBasedExampleManagement(size_t frameCapacity, size_t requiredSize) :
		examples(frameCapacity), oldestEntry(0), requiredSize(requiredSize) {}

void FrameBasedExampleManagement::add(const vector<Mat>& newExamples) {
	examples[oldestEntry].clear();
	for (const Mat& example : newExamples)
		examples[oldestEntry].push_back(example);
	++oldestEntry;
	if (oldestEntry >= examples.size())
		oldestEntry = 0;
}

void FrameBasedExampleManagement::clear() {
	for (vector<Mat>& frame : examples)
		frame.clear();
	oldestEntry = 0;
}

size_t FrameBasedExampleManagement::size() const {
	size_t count = 0;
	for (const vector<Mat>& frame : examples)
		count += frame.size();
	return count;
}

bool FrameBasedExampleManagement::hasRequiredSize() const {
	return size() >= requiredSize;
}

unique_ptr<ExampleManagement::ExampleIterator> FrameBasedExampleManagement::iterator() const {
	return unique_ptr<FrameIterator>(new FrameIterator(examples));
}

FrameBasedExampleManagement::FrameIterator::FrameIterator(const vector<vector<Mat>>& examples) :
		currentFrame(examples.cbegin()), endFrame(examples.cend()), current(currentFrame->begin()), end(currentFrame->end()) {}

bool FrameBasedExampleManagement::FrameIterator::hasNext() const {
	return current != end || currentFrame != endFrame;
}

const Mat& FrameBasedExampleManagement::FrameIterator::next() {
	if (current == end) {
		++currentFrame;
		current = currentFrame->begin();
		end = currentFrame->end();
	}
	const Mat& example = *current;
	++current;
	return example;
}

} /* namespace classification */

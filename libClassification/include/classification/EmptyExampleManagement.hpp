/*
 * EmptyExampleManagement.hpp
 *
 *  Created on: 28.11.2013
 *      Author: poschmann
 */

#ifndef EMPTYEXAMPLEMANAGEMENT_HPP_
#define EMPTYEXAMPLEMANAGEMENT_HPP_

#include "classification/ExampleManagement.hpp"
#include <stdexcept>

namespace classification {

/**
 * Examples storage with a size of zero. Will never store any training examples and state that there are
 * enough examples for training.
 */
class EmptyExampleManagement : public ExampleManagement {
public:

	/**
	 * Constructs a new empty example management.
	 */
	EmptyExampleManagement() {}

	void add(const std::vector<cv::Mat>& newExamples) {}

	void clear() {}

	size_t size() const {
		return 0;
	}

	bool hasRequiredSize() const {
		return true;
	}

	std::unique_ptr<ExampleManagement::ExampleIterator> iterator() const {
		return std::unique_ptr<ExampleIterator>(new EmptyIterator());
	}

private:

	/**
	 * Example iterator that does iterate over zero elements.
	 */
	class EmptyIterator : public ExampleIterator {
	public:

		/**
		 * Constructs a new empty iterator.
		 */
		EmptyIterator() {}

		bool hasNext() const {
			return false;
		}

		const cv::Mat& next() {
			throw std::runtime_error("EmptyExampleManagement::EmptyIterator::next: there is no element to iterate over");
		}
	};
};

} /* namespace classification */
#endif /* EMPTYEXAMPLEMANAGEMENT_HPP_ */

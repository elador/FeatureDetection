/*
 * VectorBasedExampleManagement.hpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#ifndef VECTORBASEDEXAMPLEMANAGEMENT_HPP_
#define VECTORBASEDEXAMPLEMANAGEMENT_HPP_

#include "classification/ExampleManagement.hpp"

namespace classification {

/**
 * Example management that stores the training examples in a single vector.
 */
class VectorBasedExampleManagement : public ExampleManagement {
public:

	/**
	 * Constructs a new vector based example management.
	 *
	 * @param[in] capacity Maximum amount of stored training examples.
	 * @param[in] requiredSize Minimum amount of training examples required for training.
	 */
	explicit VectorBasedExampleManagement(size_t capacity, size_t requiredSize = 1);

	virtual ~VectorBasedExampleManagement();

	void clear();

	size_t size() const;

	bool hasRequiredSize() const;

	std::unique_ptr<ExampleManagement::ExampleIterator> iterator() const;

protected:

	std::vector<cv::Mat> examples; ///< Stored training examples.
	size_t requiredSize; ///< Minimum amount of training examples required for training.

private:

	/**
	 * Example iterator that iterates over the training examples inside a vector.
	 */
	class VectorIterator : public ExampleIterator {
	public:

		/**
		 * Constructs a new vector based example iterator.
		 *
		 * @param[in] examples Training examples to iterate over.
		 */
		VectorIterator(const std::vector<cv::Mat>& examples);

		bool hasNext() const;

		const cv::Mat& next();

	private:

		std::vector<cv::Mat>::const_iterator current; ///< Iterator that points to the current training example.
		std::vector<cv::Mat>::const_iterator end;     ///< Iterator that points behind the last training example.
	};
};

} /* namespace classification */
#endif /* VECTORBASEDEXAMPLEMANAGEMENT_HPP_ */

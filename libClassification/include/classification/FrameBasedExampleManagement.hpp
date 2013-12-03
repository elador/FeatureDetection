/*
 * FrameBasedExampleManagement.hpp
 *
 *  Created on: 28.11.2013
 *      Author: poschmann
 */

#ifndef FRAMEBASEDEXAMPLEMANAGEMENT_HPP_
#define FRAMEBASEDEXAMPLEMANAGEMENT_HPP_

#include "classification/ExampleManagement.hpp"

namespace classification {

/**
 * Example storage that stores the examples by frame (a call to add is considered to add the examples of a frame).
 * The examples of a certain amount of frames is stored.
 */
class FrameBasedExampleManagement : public ExampleManagement {
public:

	/**
	 * Constructs a new frame-based example management.
	 *
	 * @param[in] frameCapacity Amount of frames to store.
	 * @param[in] requiredSize Minimum amount of training examples required for training.
	 */
	explicit FrameBasedExampleManagement(size_t frameCapacity, size_t requiredSize = 1);

	void add(const std::vector<cv::Mat>& newExamples);

	void clear();

	size_t size() const;

	bool hasRequiredSize() const;

	std::unique_ptr<ExampleManagement::ExampleIterator> iterator() const;

private:

	std::vector<std::vector<cv::Mat>> examples; ///< Stored training examples, grouped by frames.
	size_t oldestEntry; ///< The index of the oldest frame.
	size_t requiredSize; ///< Minimum amount of training examples required for training.

	/**
	 * Example iterator that iterates over the frames and over the examples inside those frames.
	 */
	class FrameIterator : public ExampleIterator {
	public:

		/**
		 * Constructs a new frame based example iterator.
		 *
		 * @param[in] examples Stored training examples, grouped by frames.
		 */
		FrameIterator(const std::vector<std::vector<cv::Mat>>& examples);

		bool hasNext() const;

		const cv::Mat& next();

	private:

		std::vector<std::vector<cv::Mat>>::const_iterator currentFrame; ///< Iterator that points to the current frame.
		std::vector<std::vector<cv::Mat>>::const_iterator endFrame;     ///< Iterator that points behind the last frame.
		std::vector<cv::Mat>::const_iterator current; ///< Iterator that points to the current training example of the current frame.
		std::vector<cv::Mat>::const_iterator end;     ///< Iterator that points behind the last training example of the current frame.
	};
};

} /* namespace classification */
#endif /* FRAMEBASEDEXAMPLEMANAGEMENT_HPP_ */

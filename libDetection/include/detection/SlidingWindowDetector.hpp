/*
 * SlidingWindowDetector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#ifndef SLIDINGWINDOWDETECTOR_HPP_
#define SLIDINGWINDOWDETECTOR_HPP_

#include "detection/Detector.hpp"

namespace detection {

/**
 * Detector that runs over an image with a sliding window of a fixed size.
 */
class SlidingWindowDetector : public Detector {
public:

	/**
	 * Constructs a new detector ... .
	 *
	 * @param[in] something Bla.
	 */
	explicit SlidingWindowDetector() {}

	virtual ~SlidingWindowDetector() {}

	/**
	 * Processes the image in a sliding window fashion.
	 *
	 * @param[in] patch The image to process.
	 * @return Maybe something?
	 */
	void detect(const Mat& image) const;

};

} /* namespace detection */
#endif /* SLIDINGWINDOWDETECTOR_HPP_ */

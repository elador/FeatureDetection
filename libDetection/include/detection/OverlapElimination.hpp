/*
 * Detector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace detection {

/**
 * Detector interface. TODO This has yet to be defined, this is only a rough and stupid draft!
 * PatchClustering / PointClustering
 */
class Detector {
public:

	/**
	 * Constructs a new detector.
	 */
	Detector() {}

	virtual ~Detector() {}

	/**
	 * Detect on an image.
	 *
	 * @param[in] image The image that the detector should run on.
	 * @return Something probably.
	 */
	virtual void detect(const Mat& image) const = 0;

};

} /* namespace detection */
#endif /* DETECTOR_HPP_ */

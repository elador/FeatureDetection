/*
 * Detector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include "opencv2/core/core.hpp"
#include <vector>
#include <memory>
#include <utility>

using cv::Mat;
using std::vector;
using std::shared_ptr;
using std::pair;

namespace imageprocessing {		// Forward-declarations from another namespace
	class Patch;
	class ImagePyramid;
}
using imageprocessing::Patch;	// TODO ... is there a better way of doing this? Just include the header...?
using imageprocessing::ImagePyramid;

namespace detection {

/**
 * Detector interface. TODO This has yet to be defined, this is only a rough and stupid draft!
 * Split into a BinaryDetector and ProbabilisticDetector? Split again between a SlidingWindow 
 * on a Pyramid and a SlidingWindow on a cv::Mat? Compose a PyrSlidingWindowDet of a MatSlidingWinDet?
 * Or two detect(Mat / Pyr) functions? ...
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
	virtual vector<pair<shared_ptr<Patch>, pair<bool, double>>> detect(shared_ptr<ImagePyramid> imagePyramid) const = 0;

};

} /* namespace detection */
#endif /* DETECTOR_HPP_ */

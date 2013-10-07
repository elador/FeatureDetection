/*
 * Detector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include "opencv2/core/core.hpp"
#include <vector>
#include <memory>
#include <utility>
#include <string>

using cv::Mat;
using cv::Rect;
using std::vector;
using std::shared_ptr;
using std::pair;

namespace imageprocessing {		// Forward-declarations from another namespace
	class Patch;
	class ImagePyramid;
	class VersionedImage;
}
using imageprocessing::Patch;	// TODO ... is there a better way of doing this? Just include the header...?
using imageprocessing::ImagePyramid;
using imageprocessing::VersionedImage;

namespace detection {

class ClassifiedPatch;

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
	 * @return A list of the patches that were positively classified by the detector.
	 */
	virtual vector<shared_ptr<ClassifiedPatch>> detect(const Mat& image) = 0;

	/**
	 * Detect on an image, but only in the region where the mask is non-zero.
	 *
	 * @param[in] image The image that the detector should run on.
	 * @param[in] image The mask .
	 * @return A list of the patches that were positively classified by the detector.
	 */
	virtual vector<shared_ptr<ClassifiedPatch>> detect(const Mat& image, const Rect& roi) = 0;

	/**
	 * Detect on an image.
	 *
	 * @param[in] image The image that the detector should run on.
	 * @return A list of the patches that were positively classified by the detector.
	 */
	virtual vector<shared_ptr<ClassifiedPatch>> detect(shared_ptr<VersionedImage> image) = 0;

public: // make private if it accomplishes what I want
	std::string landmark;

};

} /* namespace detection */
#endif /* DETECTOR_HPP_ */

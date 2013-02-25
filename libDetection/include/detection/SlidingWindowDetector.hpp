/*
 * SlidingWindowDetector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#ifndef SLIDINGWINDOWDETECTOR_HPP_
#define SLIDINGWINDOWDETECTOR_HPP_

#include "detection/Detector.hpp"

namespace classification {
	class ProbabilisticClassifier;
}
using classification::ProbabilisticClassifier;

namespace detection {

/**
 * Detector that runs over an image with a sliding window of a fixed size.
 * TODO: What do we do with BinaryClassifier/ProbabilisticClassifier? Do we also make two different SlidingWindowDetectors?
 */
class SlidingWindowDetector : public Detector {
public:

	/**
	 * Constructs a new detector ... .
	 *
	 * @param[in] something Bla.
	 */
	explicit SlidingWindowDetector(shared_ptr<ProbabilisticClassifier> classifier, int stepSizeX=1, int stepSizeY=1);

	virtual ~SlidingWindowDetector() {}

	/**
	 * Processes the image in a sliding window fashion.
	 *
	 * @param[in] patch The image to process.
	 * @return TODO - Should we return all patches or only the ones that pass the classifier?
	 *				- We should return those patches and their classifier output. It should be in some kind
						of structure that it can be sorted by the OverlapElimination! vector, unordered_map, unordered_set?
						A wrapper-class around Patch? What do we do when we want to re-use Patch-data for different classifiers
						and "attach" several outputs to one Patch? Implement like in the "before-merge"-Lib ?
	 */
	vector<pair<shared_ptr<Patch>, pair<bool, double>>> detect(shared_ptr<ImagePyramid> imagePyramid) const;

private:
	shared_ptr<ProbabilisticClassifier> classifier;	///< The classifier that is used to evaluate every step of the sliding window.
	int stepSizeX;	///< The step-size in pixels which the detector should move forward in x direction in every step. Default 1.
	int stepSizeY;	///< The step-size in pixels which the detector should move forward in y direction in every step. Default 1.

};

} /* namespace detection */
#endif /* SLIDINGWINDOWDETECTOR_HPP_ */

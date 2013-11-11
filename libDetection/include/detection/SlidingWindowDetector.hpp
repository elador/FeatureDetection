/*
 * SlidingWindowDetector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef SLIDINGWINDOWDETECTOR_HPP_
#define SLIDINGWINDOWDETECTOR_HPP_

#include "detection/Detector.hpp"

namespace classification {
	class ProbabilisticClassifier;
}
using classification::ProbabilisticClassifier;

namespace imageprocessing {
	class PyramidFeatureExtractor;
}
using imageprocessing::PyramidFeatureExtractor;

namespace detection {

/**
 * Detector that runs over an image with a sliding window of a fixed size and uses a classifier to classify every patch.
 * TODO: What do we do with BinaryClassifier/ProbabilisticClassifier? Do we also make two different SlidingWindowDetectors?
 */
class SlidingWindowDetector : public Detector {
public:

	/**
	 * Constructs a new sliding window detector.
	 *
	 * @param[in] classifier The classifier that is used to classify every image patch.
	 * @param[in] featureExtractor The image pyramid based feature extractor.
	 * @param[in] stepSizeX The step-size in x-direction the detector moves forward on the pyramids in every step.
	 * @param[in] stepSizeY The step-size in y-direction the detector moves forward on the pyramids in every step.
	 */
	SlidingWindowDetector(shared_ptr<ProbabilisticClassifier> classifier, shared_ptr<PyramidFeatureExtractor> featureExtractor, int stepSizeX=1, int stepSizeY=1);

	virtual ~SlidingWindowDetector() {}

	/**
	 * Processes the image in a sliding window fashion.
	 *
	 * @param[in] image The image to process.
	 * @return TODO - Should we return all patches or only the ones that pass the classifier?
	 *				- We should return those patches and their classifier output. It should be in some kind
						of structure that it can be sorted by the OverlapElimination! vector, unordered_map, unordered_set?
						A wrapper-class around Patch? What do we do when we want to re-use Patch-data for different classifiers
						and "attach" several outputs to one Patch? Implement like in the "before-merge"-Lib ?
	 */
	vector<shared_ptr<ClassifiedPatch>> detect(const Mat& image);

	/**
	 * Processes the image in a sliding window fashion, but only in the region where the mask is non-zero.
	 *
	 * @param[in] image The image to process.
	 * @param[in] image The mask .
	 * @return TODO - Should we return all patches or only the ones that pass the classifier?
	 *				- We should return those patches and their classifier output. It should be in some kind
						of structure that it can be sorted by the OverlapElimination! vector, unordered_map, unordered_set?
						A wrapper-class around Patch? What do we do when we want to re-use Patch-data for different classifiers
						and "attach" several outputs to one Patch? Implement like in the "before-merge"-Lib ?
					Returns a vector of _sorted_ patches that got positively classified by detector.
	 */
	vector<shared_ptr<ClassifiedPatch>> detect(const Mat& image, const Rect& roi);

	/**
	 * Processes the image in a sliding window fashion.
	 *
	 * @param[in] image The image to process.
	 * @return TODO - Should we return all patches or only the ones that pass the classifier?
	 *				- We should return those patches and their classifier output. It should be in some kind
						of structure that it can be sorted by the OverlapElimination! vector, unordered_map, unordered_set?
						A wrapper-class around Patch? What do we do when we want to re-use Patch-data for different classifiers
						and "attach" several outputs to one Patch? Implement like in the "before-merge"-Lib ?
	 */
	vector<shared_ptr<ClassifiedPatch>> detect(shared_ptr<VersionedImage> image);

	/**
	 * Processes the image in a sliding window fashion and return a probability map for each scale.
	 *
	 * @param[in] image The image to process.
	 * @return A probability map for each scale.
	 */
	vector<Mat> calculateProbabilityMaps(const Mat& image);

	// Todo: I think we shouldn't expose this function, because the featureExtractor is not up-to-date, as
	// long as detect(...) is not called? Why was this needed in the first place?
	const shared_ptr<PyramidFeatureExtractor> getPyramidFeatureExtractor() const {
		return featureExtractor;
	}

private:

	/**
	 * Classifies each image patch extracted by a sliding window approach.
	 *
	 * @return TODO
	 */
	vector<shared_ptr<ClassifiedPatch>> detect() const;

	shared_ptr<ProbabilisticClassifier> classifier;	///< The classifier that is used to evaluate every step of the sliding window.
	shared_ptr<PyramidFeatureExtractor> featureExtractor;	///< The image pyramid based feature extractor.
	int stepSizeX;	///< The step-size in pixels which the detector should move forward in x direction in every step. Default 1.
	int stepSizeY;	///< The step-size in pixels which the detector should move forward in y direction in every step. Default 1.

};

} /* namespace detection */
#endif /* SLIDINGWINDOWDETECTOR_HPP_ */

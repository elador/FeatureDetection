/*
 * FiveStageSlidingWindowDetector.hpp
 *
 *  Created on: 10.05.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FIVESTAGESLIDINGWINDOWDETECTOR_HPP_
#define FIVESTAGESLIDINGWINDOWDETECTOR_HPP_

#include "detection/Detector.hpp"
#include "detection/SlidingWindowDetector.hpp"
#include "detection/OverlapElimination.hpp"
#include "classification/ProbabilisticClassifier.hpp"

namespace detection {

/**
 * Detector that runs over an image with a sliding window of a fixed size and uses a cascade of 5 stages to classify every patch.
 * The 5 stages are as follows: A fast cascaded WVM, followed by an overlap elimination, then a more accurate SVM classifier
 * and last again an overlap elimination followed by an optional reduction of the faces found to a fixed value.
 */
class FiveStageSlidingWindowDetector : public Detector {
public:

	/**
	 * Constructs a new sliding window detector that consists of 5 stages of a fast but weak classifier, 
	 * patch-clustering and a more powerful classifier.
	 *
	 * @param[in] classifier The classifier that is used to classify every image patch.
	 * @param[in] featureExtractor The image pyramid based feature extractor.
	 * @param[in] stepSizeX The step-size in x-direction the detector moves forward on the pyramids in every step.
	 * @param[in] stepSizeY The step-size in y-direction the detector moves forward on the pyramids in every step.
	 */
	FiveStageSlidingWindowDetector(shared_ptr<SlidingWindowDetector> slidingWindowDetector, shared_ptr<OverlapElimination> overlapElimination, shared_ptr<ProbabilisticClassifier> probabilisticClassifier);

	virtual ~FiveStageSlidingWindowDetector() {}

	/**
	 * Processes the image in a sliding window fashion.
	 *
	 * @param[in] image The image to process.
	 * @return A list of all the patches that passed all the stages of the classifier.
	 */
	vector<shared_ptr<ClassifiedPatch>> detect(const Mat& image);

	vector<shared_ptr<ClassifiedPatch>> detect(const Mat& image, const Rect& roi);

	/**
	 * Processes the image in a sliding window fashion.
	 *
	 * @param[in] image The image to process.
	 * @return A list of all the patches that passed all the stages of the classifier.
	 */
	vector<shared_ptr<ClassifiedPatch>> detect(shared_ptr<VersionedImage> image);

	/**
	 * Processes the image in a sliding window fashion and return a probability map for each scale.
	 *
	 * @param[in] image The image to process.
	 * @return A probability map for each scale.
	 */
	vector<Mat> calculateProbabilityMaps(const Mat& image);

	const shared_ptr<PyramidFeatureExtractor> getPyramidFeatureExtractor() const;

private:

	shared_ptr<SlidingWindowDetector> slidingWindowDetector;	///< The SlidingWindowDetector that is used as the first stage of this detector. TODO: Think about the inheritance and this.
	shared_ptr<OverlapElimination> overlapElimination;	///< The ...
	shared_ptr<ProbabilisticClassifier> strongClassifier;	///< The strong classifier used for the remaining patches in stage 3.

};

} /* namespace detection */
#endif /* FIVESTAGESLIDINGWINDOWDETECTOR_HPP_ */

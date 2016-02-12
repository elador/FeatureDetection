/*
 * AggregatedFeaturesDetector.hpp
 *
 *  Created on: 22.10.2015
 *      Author: poschmann
 */

#ifndef AGGREGATEDFEATURESDETECTOR_HPP_
#define AGGREGATEDFEATURESDETECTOR_HPP_

#include "classification/SvmClassifier.hpp"
#include "detection/SimpleDetector.hpp"
#include "detection/NonMaximumSuppression.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include <utility>
#include <vector>

namespace detection {

/**
 * Detector that is based upon aggregated features.
 */
class AggregatedFeaturesDetector : public SimpleDetector {
public:

	/**
	 * Constructs a new aggregated features detector.
	 *
	 * @param[in] imageFilter Image filter that is applied to the image before creating the image pyramid.
	 * @param[in] layerFilter Filter that computes aggregated features on images.
	 * @param[in] cellSize Width and height of the feature descriptor cells in pixels.
	 * @param[in] windowSize Detection window size in cells.
	 * @param[in] octaveLayerCount Number of layers per image pyramid octave.
	 * @param[in] svm Linear support vector machine.
	 * @param[in] nonMaximumSuppression Non-maximum suppression.
	 */
	AggregatedFeaturesDetector(std::shared_ptr<imageprocessing::ImageFilter> imageFilter,
			std::shared_ptr<imageprocessing::ImageFilter> layerFilter, int cellSize, cv::Size windowSize, int octaveLayerCount,
			std::shared_ptr<classification::SvmClassifier> svm, std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression);

	/**
	 * Constructs a new aggregated features detector.
	 *
	 * @param[in] filter Filter that computes aggregated features on images.
	 * @param[in] cellSize Width and height of the feature descriptor cells in pixels.
	 * @param[in] windowSize Detection window size in cells.
	 * @param[in] octaveLayerCount Number of layers per image pyramid octave.
	 * @param[in] svm Linear support vector machine.
	 * @param[in] nonMaximumSuppression Non-maximum suppression.
	 */
	AggregatedFeaturesDetector(std::shared_ptr<imageprocessing::ImageFilter> filter, int cellSize, cv::Size windowSize, int octaveLayerCount,
			std::shared_ptr<classification::SvmClassifier> svm, std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression);

	/**
	 * Constructs a new aggregated features detector.
	 *
	 * @param[in] featureExtractor Aggregated features extractor.
	 * @param[in] svm Linear support vector machine.
	 * @param[in] nonMaximumSuppression Non-maximum suppression.
	 */
	AggregatedFeaturesDetector(std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor,
			std::shared_ptr<classification::SvmClassifier> svm, std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression);

	using SimpleDetector::detect;

	std::vector<cv::Rect> detect(std::shared_ptr<imageprocessing::VersionedImage> image) override;

	/**
	 * @return SVM score threshold that must be overcome for windows to be considered positive.
	 */
	float getScoreThreshold() const;

	/**
	 * @param[in] threshold SVM score threshold that must be overcome for windows to be considered positive.
	 */
	void setScoreThreshold(float threshold);

private:

	/**
	 * Updates the score pyramid for detection of targets inside a new image.
	 *
	 * @param[in] image New image.
	 */
	void update(std::shared_ptr<imageprocessing::VersionedImage> image);

	/**
	 * Determines the position of targets using the score pyramid.
	 *
	 * @return Bounding boxes around the detected targets.
	 */
	std::vector<cv::Rect> detect();

	/**
	 * Searches the score pyramid for positive values to find all possible target candidates.
	 *
	 * @return Positive windows with their SVM score.
	 */
	std::vector<Detection> getPositiveWindows();

	/**
	 * Extracts the bounding boxes from the given detections.
	 *
	 * @param[in] detections Detected targets with their score.
	 * @return Bounding boxes around the targets.
	 */
	std::vector<cv::Rect> extractBoundingBoxes(std::vector<Detection> detections);

	std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor;
	std::shared_ptr<imageprocessing::ImagePyramid> scorePyramid; ///< Classification score pyramid.
	std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression;
	cv::Size kernelSize;
	float scoreThreshold; ///< SVM score threshold that must be overcome for windows to be considered positive.
};

} /* namespace detection */

#endif /* AGGREGATEDFEATURESDETECTOR_HPP_ */

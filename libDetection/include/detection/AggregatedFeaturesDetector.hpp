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
	 * @param[in] widthScale Scaling factor to compute the actual bounding box width from positively classified windows.
	 * @param[in] heightScale Scaling factor to compute the actual bounding box height from positively classified windows.
	 * @param[in] minWindowWidth Width of the smallest detectable window in pixels (cannot be smaller than actual window width in pixels).
	 */
	AggregatedFeaturesDetector(std::shared_ptr<imageprocessing::ImageFilter> imageFilter,
			std::shared_ptr<imageprocessing::ImageFilter> layerFilter, int cellSize, cv::Size windowSize, int octaveLayerCount,
			std::shared_ptr<classification::SvmClassifier> svm, std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression,
			float widthScale = 1.0f, float heightScale = 1.0f, int minWindowWidth = 0);

	/**
	 * Constructs a new aggregated features detector.
	 *
	 * @param[in] filter Filter that computes aggregated features on images.
	 * @param[in] cellSize Width and height of the feature descriptor cells in pixels.
	 * @param[in] windowSize Detection window size in cells.
	 * @param[in] octaveLayerCount Number of layers per image pyramid octave.
	 * @param[in] svm Linear support vector machine.
	 * @param[in] nonMaximumSuppression Non-maximum suppression.
	 * @param[in] widthScale Scaling factor to compute the actual bounding box width from positively classified windows.
	 * @param[in] heightScale Scaling factor to compute the actual bounding box height from positively classified windows.
	 * @param[in] minWindowWidth Width of the smallest detectable window in pixels (cannot be smaller than actual window width in pixels).
	 */
	AggregatedFeaturesDetector(std::shared_ptr<imageprocessing::ImageFilter> filter, int cellSize, cv::Size windowSize, int octaveLayerCount,
			std::shared_ptr<classification::SvmClassifier> svm, std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression,
			float widthScale = 1.0f, float heightScale = 1.0f, int minWindowWidth = 0);

	/**
	 * Constructs a new aggregated features detector.
	 *
	 * @param[in] featureExtractor Aggregated features extractor.
	 * @param[in] svm Linear support vector machine.
	 * @param[in] nonMaximumSuppression Non-maximum suppression.
	 * @param[in] widthScale Scaling factor to compute the actual bounding box width from positively classified windows.
	 * @param[in] heightScale Scaling factor to compute the actual bounding box height from positively classified windows.
	 */
	AggregatedFeaturesDetector(std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor,
			std::shared_ptr<classification::SvmClassifier> svm, std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression,
			float widthScale = 1.0f, float heightScale = 1.0f);

	using SimpleDetector::detect;

	std::vector<cv::Rect> detect(std::shared_ptr<imageprocessing::VersionedImage> image) override;

	using SimpleDetector::detectWithScores;

	std::vector<std::pair<cv::Rect, float>> detectWithScores(std::shared_ptr<imageprocessing::VersionedImage> image) override;

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
	 * Determines the position and score of targets using the score pyramid.
	 *
	 * @return Bounding boxes around the detected targets with their scores, ordered by score in descending order.
	 */
	std::vector<std::pair<cv::Rect, float>> detectWithScores();

	/**
	 * Searches the score pyramid for positive values to find all possible target candidates.
	 *
	 * @return Positive windows with their SVM score.
	 */
	std::vector<Detection> getPositiveWindows();

	/**
	 * Rescales a positively classified window to the actual bounding box size.
	 *
	 * @param[in] boundingBox Positively classified window.
	 * @return Rescaled bounding box.
	 */
	cv::Rect rescaleWindow(cv::Rect bounds) const;

	/**
	 * Extracts the bounding boxes from the given detections.
	 *
	 * @param[in] detections Detected targets with their score.
	 * @return Bounding boxes around the targets.
	 */
	std::vector<cv::Rect> extractBoundingBoxes(std::vector<Detection> detections);

	/**
	 * Extracts the bounding boxes and scores from the given detections.
	 *
	 * @param[in] detections Detected targets with their score.
	 * @return Bounding boxes around the targets with their scores, ordered by score in descending order.
	 */
	std::vector<std::pair<cv::Rect, float>> extractBoundingBoxesWithScores(std::vector<Detection> detections);

	std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor;
	std::shared_ptr<imageprocessing::ImagePyramid> scorePyramid; ///< Classification score pyramid.
	std::shared_ptr<detection::NonMaximumSuppression> nonMaximumSuppression;
	cv::Size kernelSize;
	float scoreThreshold; ///< SVM score threshold that must be overcome for windows to be considered positive.
	float widthScale; ///< Scaling factor to compute the actual bounding box width from positively classified windows.
	float heightScale; ///< Scaling factor to compute the actual bounding box height from positively classified windows.
};

} /* namespace detection */

#endif /* AGGREGATEDFEATURESDETECTOR_HPP_ */

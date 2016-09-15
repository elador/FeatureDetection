/*
 * DetectorTrainer.hpp
 *
 *  Created on: 21.10.2015
 *      Author: poschmann
 */

#ifndef DETECTORTRAINER_HPP_
#define DETECTORTRAINER_HPP_

#include "LabeledImage.hpp"
#include "classification/ConfidenceBasedExampleManagement.hpp"
#include "detection/AggregatedFeaturesDetector.hpp"
#include "detection/NonMaximumSuppression.hpp"
#include "libsvm/LibSvmClassifier.hpp"
#include "imageio/RectLandmark.hpp"
#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "opencv2/core/core.hpp"
#include <memory>
#include <random>
#include <string>
#include <vector>

/**
 * Parameters of the features.
 */
struct FeatureParams {
	cv::Size windowSizeInCells; ///< Detection window size in cells.
	int cellSizeInPixels; ///< Cell size in pixels.
	int octaveLayerCount; ///< Number of image pyramid layers per octave.
	float widthScaleFactor = 1.0f; ///< Scale factor that is applied to the window width before training.
	float heightScaleFactor = 1.0f; ///< Scale factor that is applied to the window height before training.

	/**
	 * @return Detection window size in pixels.
	 */
	cv::Size windowSizeInPixels() const {
		return cv::Size(
				windowSizeInCells.width * cellSizeInPixels,
				windowSizeInCells.height * cellSizeInPixels
		);
	}

	/**
	 * @return Aspect ratio of the window (width / height).
	 */
	double windowAspectRatio() const {
		return static_cast<double>(windowSizeInCells.width) / static_cast<double>(windowSizeInCells.height);
	}

	/**
	 * @return Scale factor that is applied to the window width after detection.
	 */
	float widthScaleFactorInv() const {
		return 1.0f / widthScaleFactor;
	}

	/**
	 * @return Scale factor that is applied to the window height after detection.
	 */
	float heightScaleFactorInv() const {
		return 1.0f / heightScaleFactor;
	}
};

/**
 * Parameters of the training example gathering and SVM training.
 */
struct TrainingParams {
	bool mirrorTrainingData = true; ///< Flag that indicates whether to horizontally mirror the training data.
	int maxNegatives = 0; ///< Maximum number of negative training examples (0 if not constrained).
	int randomNegativesPerImage = 20; ///< Number of initial random negatives per image.
	int maxHardNegativesPerImage = 100; ///< Maximum number of hard negatives per image and bootstrapping round.
	int bootstrappingRounds = 3; ///< Number of bootstrapping rounds.
	float negativeScoreThreshold = -1.0f; ///< SVM score threshold for retrieving strong negative examples.
	double overlapThreshold = 0.3; ///< Maximum allowed overlap between negative examples and non-negative annotations.
	double C = 1;
	bool compensateImbalance = false; ///< Flag that indicates whether to adjust class weights to compensate for unbalanced data.
	bool probabilistic = false; ///< Flag that indicates whether to compute logistic function parameters for probabilistic output.
};

/**
 * Management of negative training examples that only keeps the hardest examples.
 */
class HardNegativeExampleManagement : public classification::ConfidenceBasedExampleManagement {
public:

	HardNegativeExampleManagement(const std::shared_ptr<classification::BinaryClassifier>& classifier, size_t capacity) :
			classification::ConfidenceBasedExampleManagement(classifier, false, capacity) {
		setFirstExamplesToKeep(0);
	}
};

/**
 * Trainer for detectors based on aggregated features and linear SVMs.
 */
class DetectorTrainer {
public:

	explicit DetectorTrainer(bool printProgressInformation = false, std::string printPrefix = "");

	/**
	 * Sets some general training parameters.
	 */
	void setTrainingParameters(TrainingParams params);

	/**
	 * Defines the features used for detection.
	 *
	 * @param[in] params Common feature parameters.
	 * @param[in] filter Image filter that transforms the image into a descriptor image, were each pixel describes a cell.
	 * @param[in] imageFilter Image filter that is applied to the image before creating the image pyramid. Optional.
	 */
	void setFeatures(FeatureParams params, const std::shared_ptr<imageprocessing::ImageFilter>& filter,
			const std::shared_ptr<imageprocessing::ImageFilter>& imageFilter = std::shared_ptr<imageprocessing::ImageFilter>());

	/**
	 * Trains the classifier that is used by the detector.
	 *
	 * The labeled images contain positive examples and fuzzy ones that will be ignored for training. Fuzzy training
	 * examples must have a name that starts with "ignore". Bounding boxes with other names are considered positive.
	 *
	 * @param[in] images Images labeled with bounding boxes around positive and fuzzy examples (anything else is considered negative).
	 */
	void train(std::vector<LabeledImage> images);

	/**
	 * Stores the SVM data into a file.
	 *
	 * @param[in] filename Name of the file.
	 */
	void storeClassifier(const std::string& filename) const;

	/**
	 * @return Weight vector of the SVM.
	 */
	cv::Mat getWeightVector() const;

	/**
	 * Creates a new detector that uses the trained classifier.
	 *
	 * @param[in] nms Non-maximum suppression algorithm.
	 */
	std::shared_ptr<detection::AggregatedFeaturesDetector> getDetector(
			std::shared_ptr<detection::NonMaximumSuppression> nms) const;

	/**
	 * Creates a new detector that uses the trained classifier, but uses a different number of pyramid layers per octave
	 * than was used for training.
	 *
	 * @param[in] nms Non-maximum suppression algorithm.
	 * @param[in] octaveLayerCount Number of image pyramid layers per octave.
	 * @param[in] threshold SVM score threshold, defaults to zero.
	 */
	std::shared_ptr<detection::AggregatedFeaturesDetector> getDetector(
			std::shared_ptr<detection::NonMaximumSuppression> nms, int octaveLayerCount, float threshold = 0) const;

private:

	void createEmptyClassifier();

	void collectInitialTrainingExamples(std::vector<LabeledImage> images);

	void collectHardTrainingExamples(std::vector<LabeledImage> images);

	void createHardNegativesDetector();

	void collectTrainingExamples(std::vector<LabeledImage> images, bool initial);

	/**
	 * Adjusts the size and aspect ratio of the landmarks to fit the feature window size.
	 *
	 * @param[in] landmarks Labeled bounding boxes with potentially differing aspect ratios.
	 * @return Labeled bounding boxes with the correct aspect ratio.
	 */
	std::vector<imageio::RectLandmark> adjustSizes(const std::vector<imageio::RectLandmark>& landmarks) const;

	/**
	 * Adjusts the size and aspect ratio of a landmark to fit the feature window size.
	 *
	 * @param[in] landmark Labeled bounding box with potentially differing aspect ratio.
	 * @return Labeled bounding box with the correct aspect ratio.
	 */
	imageio::RectLandmark adjustSize(const imageio::RectLandmark& landmark) const;

	void addMirroredTrainingExamples(const cv::Mat& image, const std::vector<imageio::RectLandmark>& landmarks, bool initial);

	cv::Mat flipHorizontally(const cv::Mat& image);

	std::vector<imageio::RectLandmark> flipHorizontally(const std::vector<imageio::RectLandmark>& landmarks, int imageWidth);

	imageio::RectLandmark flipHorizontally(const imageio::RectLandmark& landmark, int imageWidth);

	void addTrainingExamples(const cv::Mat& image, const std::vector<imageio::RectLandmark>& landmarks, bool initial);

	void addTrainingExamples(const cv::Mat& image, const Annotations& annotations, bool initial);

	void setImage(const cv::Mat& image);

	void addPositiveExamples(const std::vector<cv::Rect>& positiveBoxes);

	void addRandomNegativeExamples(const std::vector<cv::Rect>& nonNegativeBoxes);

	cv::Rect createRandomBounds() const;

	void addHardNegativeExamples(const std::vector<cv::Rect>& nonNegativeBoxes);

	bool addNegativeIfNotOverlapping(cv::Rect candidate, const std::vector<cv::Rect>& nonNegativeBoxes);

	bool isOverlapping(cv::Rect boxToTest, const std::vector<cv::Rect>& otherBoxes) const;

	double computeOverlap(cv::Rect a, cv::Rect b) const;

	void trainClassifier();

	void retrainClassifier();

	void trainClassifier(bool initial);

	bool printProgressInformation;
	std::string printPrefix;
	TrainingParams trainingParams;
	std::shared_ptr<detection::NonMaximumSuppression> noSuppression;
	FeatureParams featureParams;
	double aspectRatio;
	double aspectRatioInv;
	std::shared_ptr<imageprocessing::ImageFilter> imageFilter;
	std::shared_ptr<imageprocessing::ImageFilter> filter;
	std::shared_ptr<imageprocessing::extraction::AggregatedFeaturesExtractor> featureExtractor;
	std::shared_ptr<libsvm::LibSvmClassifier> classifier;
	std::shared_ptr<detection::AggregatedFeaturesDetector> hardNegativesDetector;
	std::vector<cv::Mat> positiveTrainingExamples;
	std::vector<cv::Mat> negativeTrainingExamples;
	cv::Mat image;
	cv::Size imageSize;
	mutable std::mt19937 generator;
};

#endif /* DETECTORTRAINER_HPP_ */

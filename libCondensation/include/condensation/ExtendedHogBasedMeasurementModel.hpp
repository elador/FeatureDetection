/*
 * ExtendedHogBasedMeasurementModel.hpp
 *
 *  Created on: 13.01.2014
 *      Author: poschmann
 */

#ifndef EXTENDEDHOGBASEDMEASUREMENTMODEL_HPP_
#define EXTENDEDHOGBASEDMEASUREMENTMODEL_HPP_

#include "condensation/AdaptiveMeasurementModel.hpp"
#include "condensation/StateValidator.hpp"
#include "opencv2/core/core.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/variate_generator.hpp"
#include <utility>
#include <forward_list>
#include <unordered_map> // only necessary for output of learned positive patches

namespace imageprocessing {
class ConvolutionFilter;
class ImagePyramid;
class FeatureExtractor;
class DirectPyramidFeatureExtractor;
class GrayscaleFilter;
class ExtendedHogFilter;
class CompleteExtendedHogFilter;
class ExtendedHogFeatureExtractor;
}

namespace classification {
class ProbabilisticSvmClassifier;
class TrainableProbabilisticSvmClassifier;
class BinaryClassifier;
class ProbabilisticClassifier;
}

namespace condensation {

/**
 * Measurement model that is based on extended HOG features and a linear SVM.
 */
class ExtendedHogBasedMeasurementModel : public AdaptiveMeasurementModel, public StateValidator {
public:

	/**
	 * Adaptation method describing where to get the positive training examples from.
	 */
	enum class Adaptation { NONE, POSITION, TRAJECTORY, CORRECTED_TRAJECTORY };

	/**
	 * Constructs a new extended HOG based measurement model.
	 *
	 * @param[in] classifier The trainable SVM classifier.
	 */
	explicit ExtendedHogBasedMeasurementModel(std::shared_ptr<classification::TrainableProbabilisticSvmClassifier> classifier);

	/**
	 * Constructs a new extended HOG based measurement model that uses the given image pyramid for the feature extraction.
	 *
	 * @param[in] classifier The trainable SVM classifier.
	 * @param[in] basePyramid Grayscale image pyramid that is used as the base for feature extraction.
	 */
	ExtendedHogBasedMeasurementModel(std::shared_ptr<classification::TrainableProbabilisticSvmClassifier> classifier,
			std::shared_ptr<imageprocessing::ImagePyramid> basePyramid);

	void update(std::shared_ptr<imageprocessing::VersionedImage> image);

	using MeasurementModel::evaluate;

	void evaluate(std::shared_ptr<imageprocessing::VersionedImage> image, std::vector<std::shared_ptr<Sample>>& samples);

	void evaluate(Sample& sample) const;

	bool isValid(const Sample& target, const std::vector<std::shared_ptr<Sample>>& samples,
			std::shared_ptr<imageprocessing::VersionedImage> image);

	bool isUsable() const;

	bool initialize(std::shared_ptr<imageprocessing::VersionedImage> image, Sample& target);

	bool adapt(std::shared_ptr<imageprocessing::VersionedImage> image, const std::vector<std::shared_ptr<Sample>>& samples, const Sample& target);

	bool adapt(std::shared_ptr<imageprocessing::VersionedImage> image, const std::vector<std::shared_ptr<Sample>>& samples);

	void reset();

	/**
	 * Returns the positive examples that were actually used for training.
	 *
	 * @return Map that associates frame indices with a bounding box that was used for training.
	 */
	const std::unordered_map<size_t, cv::Rect>& getLearned() const;

	/**
	 * Changes the parameters for the extraction of the extended HOG features.
	 *
	 * @param[in] cellSize Width and height of the HOG cells in pixels.
	 * @param[in] cellCount Preferred HOG cell count (actual count might deviate depending on aspect ratio).
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] interpolateBins Flag that indicates whether a gradient should contribute to two neighboring bins in a weighted manner.
	 * @param[in] interpolateCells Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] octaveLayerCount Number of layers per image pyramid octave.
	 */
	void setHogParams(size_t cellSize, size_t cellCount, bool signedAndUnsigned = false,
			bool interpolateBins = false, bool interpolateCells = true, int octaveLayerCount = 5);

	/**
	 * Changes the score threshold for rejecting samples (setting their target-flag to false and invalidating
	 * the target state).
	 *
	 * @param[in] rejectionThreshold Score threshold for state and sample rejection.
	 */
	void setRejectionThreshold(double rejectionThreshold);

	/**
	 * Changes whether a sliding window approach for particle evaluation should be used.
	 *
	 * @param[in] useSlidingWindow Flag that indicates whether a sliding window approach for particle evaluation should be used.
	 * @param[in] conservativeReInit Flag that indicates whether the peak score also has to overcome the adaptation threshold before re-initializing (only considered when sliding window approach is used).
	 */
	void setUseSlidingWindow(bool useSlidingWindow, bool conservativeReInit = false);

	/**
	 * Changes the parameters for the selection of negative training examples.
	 *
	 * @param[in] negativeExampleCount Maximum number of negative examples used for re-training.
	 * @param[in] initialNegativeExampleCount Number of negative examples used for the initial training.
	 * @param[in] randomExampleCount Number of randomly generated examples for choosing the negative examples from (only used without sliding window).
	 * @param[in] negativeScoreThreshold Score threshold for negative training examples.
	 */
	void setNegativeExampleParams(size_t negativeExampleCount, size_t initialNegativeExampleCount,
			size_t randomExampleCount, float negativeScoreThreshold = -1.0f);

	/**
	 * Changes the overlap thresholds of the negative training examples.
	 *
	 * @param[in] positiveOverlapThreshold Allowed overlap of negative training examples with the estimated target.
	 * @param[in] negativeOverlapThreshold Allowed overlap of negative training examples with other negative training examples.
	 */
	void setOverlapThresholds(double positiveOverlapThreshold, double negativeOverlapThreshold);

	/**
	 * Changes the adaptation strategy.
	 *
	 * @param[in] adaptation Adpatation strategy.
	 * @param[in] adaptationThreshold Threshold of the target SVM score for re-training.
	 * @param[in] exclusionThreshold SVM score threshold for adding positive training examples from the trajectory (only used for trajectory learning).
	 */
	void setAdaptation(Adaptation adaptation, double adaptationThreshold = 0.75, double exclusionThreshold = 0.0);

private:

	/**
	 * Retrieves the peak of the heat map.
	 *
	 * @return A pair containing the highest heat value and its bounding box.
	 */
	std::pair<double, cv::Rect> getHeatPeak() const;

	/**
	 * Computes the weighted mean of the given samples.
	 *
	 * @param[in] samples The samples.
	 * @param[in] weights The weight for each sample.
	 * @return The weighted mean.
	 */
	std::shared_ptr<Sample> getMean(std::vector<std::shared_ptr<Sample>> samples, std::vector<double> weights) const;

	/**
	 * Creates positive training examples.
	 *
	 * @param[in] samples The weighted samples.
	 * @param[in] target The estimated target position.
	 * @return The positive training examples. Might be empty if there should be no training at all.
	 */
	std::vector<cv::Mat> createPositiveTrainingExamples(const std::vector<std::shared_ptr<Sample>>& samples, const Sample& target);

	/**
	 * Creates the negative training examples.
	 *
	 * @param[in] image The image.
	 * @param[in] target The estimated target position.
	 * @return The negative training examples.
	 */
	std::vector<cv::Mat> createNegativeTrainingExamples(const cv::Mat& image, const Sample& target) const;

	/**
	 * Selects negative training examples from the background of the image. It is assumed that this model
	 * was updated with the image before calling this function.
	 *
	 * @param[in] targetBounds The bounding box of the target (single positive example).
	 * @param[in] allowedOverlap The maximum allowed overlap between the target and negative examples.
	 * @return The good negative training examples.
	 */
	std::vector<cv::Mat> createGoodNegativeExamples(cv::Rect targetBounds) const;

	/**
	 * Randomly samples negative (training/testing) examples from the image. It is assumed that this model
	 * was updated with the given image before calling this function.
	 *
	 * @param[in] count The amount of negative examples to create.
	 * @param[in] image The image.
	 * @param[in] targetBounds The bounding box of the target (single positive example).
	 * @param[in] allowedOverlap The maximum allowed overlap between the target and negative examples.
	 * @return The random negative examples.
	 */
	std::vector<cv::Mat> createRandomNegativeExamples(
			size_t count, const cv::Mat& image, cv::Rect targetBounds) const;

	/**
	 * Randomly samples a bounding box that is completely within the given image dimensions.
	 *
	 * @param[in] image The image.
	 * @return The random bounding box.
	 */
	cv::Rect createRandomBounds(const cv::Mat& image) const;

	/**
	 * Computes the overlap between two rectangles.
	 *
	 * @param[in] a The first rectangle.
	 * @param[in] b The second rectangle.
	 * @return The overlap percentage between the two rectangles.
	 */
	double computeOverlap(cv::Rect a, cv::Rect b) const;

	size_t cellSize; ///< Width and height of the HOG cells in pixels.
	size_t cellCount; ///< Preferred HOG cell count (actual count might deviate depending on aspect ratio).
	bool signedAndUnsigned; ///< Flag that indicates whether signed and unsigned gradients should be used.
	bool interpolateBins; ///< Flag that indicates whether a gradient should contribute to two neighboring bins in a weighted manner.
	bool interpolateCells; ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	int octaveLayerCount; ///< Number of layers per image pyramid octave.
	double rejectionThreshold; ///< Score threshold for state and sample rejection.
	bool useSlidingWindow; ///< Flag that indicates whether a sliding window approach for particle evaluation should be used.
	bool conservativeReInit; ///< Flag that indicates whether the peak score also has to overcome the adaptation threshold before re-initializing (only considered when sliding window approach is used).
	size_t negativeExampleCount; ///< Maximum number of negative examples used for re-training.
	size_t initialNegativeExampleCount; ///< Number of negative examples used for the initial training.
	size_t randomExampleCount; ///< Number of randomly generated examples for choosing the negative examples from (only used without sliding window).
	float negativeScoreThreshold; ///< Score threshold for negative training examples.
	double positiveOverlapThreshold; ///< Allowed overlap of negative training examples with the positive training example.
	double negativeOverlapThreshold; ///< Allowed overlap of negative training examples with other negative training examples.
	Adaptation adaptation; ///< Adaptation strategy.
	double adaptationThreshold; ///< Threshold of the target SVM score for re-training.
	double exclusionThreshold; ///< SVM score threshold for adding positive training examples from the trajectory (only used for trajectory learning).
	std::shared_ptr<imageprocessing::ConvolutionFilter> convolutionFilter; ///< Filter for computing the SVM scores over the HOG cell image.
	std::shared_ptr<imageprocessing::ImagePyramid> basePyramid; ///< Grayscale image pyramid.
	std::shared_ptr<imageprocessing::ImagePyramid> heatPyramid; ///< Image pyramid containing the SVM scores of each location.
	std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor; ///< Extractor of the extended HOG features.
	std::shared_ptr<imageprocessing::ExtendedHogFeatureExtractor> positiveFeatureExtractor; ///< Extractor of the extended HOG features of positive examples.
	std::shared_ptr<imageprocessing::FeatureExtractor> heatExtractor; ///< Extractor of the SVM scores.
	std::shared_ptr<classification::ProbabilisticSvmClassifier> classifier; ///< SVM classifier for computing the particle weights.
	std::shared_ptr<classification::TrainableProbabilisticSvmClassifier> trainable; ///< SVM classifier for re-training.
	size_t cellRowCount; ///< HOG cell row count.
	size_t cellColumnCount; ///< HOG cell column count.
	size_t minWidth; ///< Largest possible width of the patch for extracting features.
	size_t maxWidth; ///< Smallest possible width of the patch for extracting features.
	bool initialized; ///< Flag that indicates whether this model was initialized.
	bool usable; ///< Flag that indicates whether this model is usable.
	bool targetLost; ///< Flag that indicates whether the target is lost.
	mutable boost::mt19937 generator; ///< Random number generator.
	boost::uniform_int<> uniformIntDistribution; ///< Uniform integer distribution.
	boost::normal_distribution<> normalDistribution; ///< Normal distribution.
	cv::Mat initialFeatures; ///< Initial positive training example.
	std::vector<cv::Mat> trajectoryFeatures; ///< Positive training examples from the current trajectory (only used for un-corrected trajectory learning).
	std::forward_list<std::shared_ptr<imageprocessing::FeatureExtractor>> pastFeatureExtractors; ///< Feature extractors of past frames (used for corrected trajectory learning).

	std::vector<std::pair<int, cv::Rect>> trajectoryToLearn; // only necessary for output of learned positive patches
	mutable size_t frameIndex; // only necessary for output of learned positive patches
	mutable std::unordered_map<size_t, cv::Rect> learned; // only necessary for output of learned positive patches
};

} /* namespace condensation */
#endif /* EXTENDEDHOGBASEDMEASUREMENTMODEL_HPP_ */

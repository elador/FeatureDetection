/*
 * CondensationTracker.hpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#ifndef CONDENSATIONTRACKER_HPP_
#define CONDENSATIONTRACKER_HPP_

#include "condensation/Sample.hpp"
#include "opencv2/core/core.hpp"
#include "boost/optional.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_real.hpp"
#include <memory>
#include <vector>

namespace imageprocessing {
class VersionedImage;
class DirectPyramidFeatureExtractor;
}

namespace classification {
class ProbabilisticClassifier;
}

namespace condensation {

class TransitionModel;
class MeasurementModel;
class ResamplingAlgorithm;
class StateExtractor;
class StateValidator;

/**
 * Condensation tracker that adapts to the appearance of the tracked target over time.
 *
 * This condensation tracker uses a measurement model that adapts to the found target. Must be initialized before
 * being used. May be initialized several times, as it may need more than one frame with the corresponding target
 * position in order to work correctly.
 */
class CondensationTracker {
public:

	/**
	 * Constructs a new adaptive condensation tracker.
	 *
	 * @param[in] transitionModel The transition model.
	 * @param[in] measurementModel The measurement model.
	 * @param[in] resamplingAlgorithm The resampling algorithm.
	 * @param[in] extractor The state extractor.
	 * @param[in] count The number of samples.
	 * @param[in] randomRate The percentage of samples that should be equally distributed across the image.
	 * @param[in] minSize The minimum size of a sample.
	 * @param[in] maxSize The maximum size of a sample.
	 */
	CondensationTracker(
			std::shared_ptr<TransitionModel> transitionModel, std::shared_ptr<MeasurementModel> measurementModel,
			std::shared_ptr<ResamplingAlgorithm> resamplingAlgorithm, std::shared_ptr<StateExtractor> extractor,
			size_t count, double randomRate, int minSize, int maxSize);

	/**
	 * Initializes this tracker at the given position. May need several subsequent initializations before
	 * being usable.
	 *
	 * @param[in] image The current image.
	 * @param[in] position The current position of the target that should be tracked.
	 * @return The bounding box around the initial target position if the tracker is usable, none otherwise.
	 */
	boost::optional<cv::Rect> initialize(const cv::Mat& image, const cv::Rect& position);

	/**
	 * Processes the next image and returns the most probable object position.
	 *
	 * @param[in] image The next image.
	 * @return The bounding box around the most probable target position if found, none otherwise.
	 */
	boost::optional<cv::Rect> process(const cv::Mat& image);

	/**
	 * Resets this tracker to its uninitialized state, so it has to be initialized again.
	 */
	void reset();

	/**
	 * @return True if the tracker has adapted to the current appearance, false otherwise.
	 */
	bool hasAdapted();

	/**
	 * @return The estimated target state.
	 */
	std::shared_ptr<Sample> getState();

	/**
	 * @return The current samples.
	 */
	const std::vector<std::shared_ptr<Sample>>& getSamples() const;

	/**
	 * Adds a validator.
	 *
	 * @param[in] validator The new target state validator.
	 */
	void addValidator(std::shared_ptr<StateValidator> validator);

	/**
	 * @return The number of samples.
	 */
	size_t getCount();

	/**
	 * @param[in] The new number of samples.
	 */
	void setCount(size_t count);

	/**
	 * @return The percentage of samples that should be equally distributed.
	 */
	double getRandomRate();

	/**
	 * @param[in] The new percentage of samples that should be equally distributed.
	 */
	void setRandomRate(double randomRate);

private:

	/**
	 * Randomly samples new values for a sample.
	 *
	 * @param[in,out] sample The sample.
	 * @param[in] image The image.
	 */
	void sampleValues(Sample& sample, const cv::Mat& image);

	size_t count; ///< The number of samples.
	double randomRate; ///< The percentage of samples that should be equally distributed.
	int minSize; ///< The minimum size of a sample.
	int maxSize; ///< The maximum size of a sample.

	std::vector<std::shared_ptr<Sample>> samples;    ///< The current samples.
	std::vector<std::shared_ptr<Sample>> oldSamples; ///< The previous samples.
	std::shared_ptr<Sample> state;                   ///< The estimated target state.
	bool adapted; ///< Flag that indicates whether the tracker has adapted to the current appearance.

	std::shared_ptr<imageprocessing::VersionedImage> image;   ///< The image used for evaluation.
	std::shared_ptr<TransitionModel> transitionModel;         ///< The transition model.
	std::shared_ptr<MeasurementModel> measurementModel;       ///< The measurement model.
	std::shared_ptr<ResamplingAlgorithm> resamplingAlgorithm; ///< The resampling algorithm.
	std::shared_ptr<StateExtractor> extractor;                ///< The state extractor.
	std::vector<std::shared_ptr<StateValidator>> validators;  ///< The validators of the target position.

	boost::mt19937 generator; ///< Random number generator.
	boost::uniform_int<> intDistribution;   ///< Uniform integer distribution.
	boost::uniform_real<> realDistribution; ///< Uniform real distribution.
};

} /* namespace condensation */
#endif /* CONDENSATIONTRACKER_HPP_ */

/*
 * AdaptiveCondensationTracker.hpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#ifndef ADAPTIVECONDENSATIONTRACKER_HPP_
#define ADAPTIVECONDENSATIONTRACKER_HPP_

#include "condensation/Sample.hpp"
#include "opencv2/core/core.hpp"
#include "boost/optional.hpp"
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

class Sampler;
class MeasurementModel;
class AdaptiveMeasurementModel;
class StateExtractor;
class StateValidator;

/**
 * Condensation tracker that adapts to the appearance of the tracked target over time.
 *
 * This condensation tracker uses a measurement model that adapts to the found target. Must be initialized before
 * being used. May be initialized several times, as it may need more than one frame with the corresponding target
 * position in order to work correctly.
 */
class AdaptiveCondensationTracker {
public:

	/**
	 * Constructs a new adaptive condensation tracker.
	 *
	 * @param[in] sampler The sampler.
	 * @param[in] measurementModel The adaptive measurement model.
	 * @param[in] extractor The state extractor.
	 * @param[in] initialCount The initial amount of particles.
	 */
	AdaptiveCondensationTracker(std::shared_ptr<Sampler> sampler, std::shared_ptr<AdaptiveMeasurementModel> measurementModel,
			std::shared_ptr<StateExtractor> extractor, int initialCount);

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
	 * @return The sampler.
	 */
	std::shared_ptr<Sampler> getSampler();

	/**
	 * @param[in] sampler The new sampler.
	 */
	void setSampler(std::shared_ptr<Sampler> sampler);

	/**
	 * Adds a validator.
	 *
	 * @param[in] validator The new target state validator.
	 */
	void addValidator(std::shared_ptr<StateValidator> validator);

private:

	int initialCount; ///< The initial amount of particles.
	std::vector<std::shared_ptr<Sample>> samples;    ///< The current samples.
	std::vector<std::shared_ptr<Sample>> oldSamples; ///< The previous samples.
	std::shared_ptr<Sample> state;                   ///< The estimated target state.
	bool adapted; ///< Flag that indicates whether the tracker has adapted to the current appearance.

	std::shared_ptr<imageprocessing::VersionedImage> image;     ///< The image used for evaluation.
	std::shared_ptr<Sampler> sampler;                           ///< The sampler.
	std::shared_ptr<AdaptiveMeasurementModel> measurementModel; ///< The adaptive measurement model.
	std::shared_ptr<StateExtractor> extractor;                  ///< The state extractor.
	std::vector<std::shared_ptr<StateValidator>> validators;    ///< The validators of the target position.
};

} /* namespace condensation */
#endif /* ADAPTIVECONDENSATIONTRACKER_HPP_ */

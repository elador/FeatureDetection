/*
 * PartiallyAdaptiveCondensationTracker.hpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#ifndef PARTIALLYADAPTIVECONDENSATIONTRACKER_HPP_
#define PARTIALLYADAPTIVECONDENSATIONTRACKER_HPP_

#include "condensation/Sample.hpp"
#include "opencv2/core/core.hpp"
#include "boost/optional.hpp"
#include <memory>
#include <vector>

namespace imageprocessing {
class VersionedImage;
}

namespace condensation {

class Sampler;
class MeasurementModel;
class AdaptiveMeasurementModel;
class StateExtractor;

/**
 * Condensation tracker that tries to adapt to the appearance of the tracked target over time.
 *
 * This condensation tracker makes use of two measurement models: a static one and an adaptive one. The
 * tracking starts with the static measurement model, while the adaptive one gets trained with the position
 * of the tracked target. If the adaptive model is usable, the static model gets replaced. If the object
 * disappears for a certain amount of time (or is simply not found for some reason), the static measurement
 * model may be used again.
 */
class PartiallyAdaptiveCondensationTracker {
public:

	/**
	 * Constructs a new partially adaptive condensation tracker.
	 *
	 * @param[in] sampler The sampler.
	 * @param[in] initialMeasurementModel The initial static measurement model.
	 * @param[in] measurementModel The adaptive measurement model.
	 * @param[in] extractor The state extractor.
	 */
	PartiallyAdaptiveCondensationTracker(std::shared_ptr<Sampler> sampler, std::shared_ptr<MeasurementModel> initialMeasurementModel,
			std::shared_ptr<AdaptiveMeasurementModel> measurementModel, std::shared_ptr<StateExtractor> extractor);

	/**
	 * Processes the next image and returns the most probable target position.
	 *
	 * @param[in] image The next image.
	 * @return The bounding box around the most probable target position if found, none otherwise.
	 */
	boost::optional<cv::Rect> process(const cv::Mat& image);

	/**
	 * @return The estimated target state.
	 */
	std::shared_ptr<Sample> getState() {
		return state;
	}

	/**
	 * @return The current samples.
	 */
	inline const std::vector<std::shared_ptr<Sample>>& getSamples() const {
		return samples;
	}

	/**
	 * @return The sampler.
	 */
	inline std::shared_ptr<Sampler> getSampler() {
		return sampler;
	}

	/**
	 * @param[in] sampler The new sampler.
	 */
	inline void setSampler(std::shared_ptr<Sampler> sampler) {
		this->sampler = sampler;
	}

	/**
	 * @return True if the adaptive measurement model should be used, false otherwise.
	 */
	inline bool isUsingAdaptiveModel() {
		return useAdaptiveModel;
	}

	/**
	 * @param[in] active Flag that indicates whether the adaptive measurement model should be used.
	 */
	void setUseAdaptiveModel(bool useAdaptive);

	/**
	 * @return True if the adaptive measurement model was used, false otherwise.
	 */
	inline bool wasUsingAdaptiveModel() {
		return usedAdaptiveModel;
	}

private:

	std::vector<std::shared_ptr<Sample>> samples;    ///< The current samples.
	std::vector<std::shared_ptr<Sample>> oldSamples; ///< The previous samples.
	std::shared_ptr<Sample> state;              ///< The estimated target state.

	bool useAdaptiveModel;  ///< Flag that indicates whether the adaptive measurement model should be used.
	bool usedAdaptiveModel; ///< Flag that indicates whether the adaptive measurement model was used.

	std::shared_ptr<imageprocessing::VersionedImage> image;     ///< The image used for evaluation.
	std::shared_ptr<Sampler> sampler;                           ///< The sampler.
	std::shared_ptr<MeasurementModel> initialMeasurementModel;  ///< The initial static measurement model.
	std::shared_ptr<AdaptiveMeasurementModel> measurementModel; ///< The adaptive measurement model.
	std::shared_ptr<StateExtractor> extractor;                  ///< The state extractor.
};

} /* namespace condensation */
#endif /* PARTIALLYADAPTIVECONDENSATIONTRACKER_HPP_ */

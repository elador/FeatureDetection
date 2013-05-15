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

using cv::Mat;
using cv::Rect;
using boost::optional;
using std::vector;
using std::shared_ptr;

namespace imageprocessing {
class VersionedImage;
}
using imageprocessing::VersionedImage;

namespace condensation {

class Sampler;
class MeasurementModel;
class AdaptiveMeasurementModel;
class PositionExtractor;

/**
 * Condensation tracker that tries to adapt to the appearance of the tracked object over time.
 *
 * This condensation tracker makes use of two measurement models: a static one and an adaptive one. The
 * tracking starts with the static measurement model, while the adaptive one gets trained with the position
 * of the tracked object. If the adaptive model is usable, the static model gets replaced. If the object
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
	 * @param[in] extractor The position extractor.
	 */
	PartiallyAdaptiveCondensationTracker(shared_ptr<Sampler> sampler, shared_ptr<MeasurementModel> initialMeasurementModel,
			shared_ptr<AdaptiveMeasurementModel> measurementModel, shared_ptr<PositionExtractor> extractor);

	~PartiallyAdaptiveCondensationTracker();

	/**
	 * Processes the next image and returns the most probable object position.
	 *
	 * @param[in] image The next image.
	 * @return The bounding box around the most probable object position if there is an object.
	 */
	optional<Rect> process(const Mat& image);

	/**
	 * @return The estimated state.
	 */
	optional<Sample> getState() {
		return state;
	}

	/**
	 * @return The current samples.
	 */
	inline const vector<Sample>& getSamples() const {
		return samples;
	}

	/**
	 * @return The sampler.
	 */
	inline shared_ptr<Sampler> getSampler() {
		return sampler;
	}

	/**
	 * @param[in] sampler The new sampler.
	 */
	inline void setSampler(shared_ptr<Sampler> sampler) {
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

	vector<Sample> samples;    ///< The current samples.
	vector<Sample> oldSamples; ///< The previous samples.
	optional<Sample> state;    ///< The estimated state.

	bool useAdaptiveModel;        ///< Flag that indicates whether the adaptive measurement model should be used.
	bool usedAdaptiveModel;       ///< Flag that indicates whether the adaptive measurement model was used.

	shared_ptr<VersionedImage> image;                      ///< The image used for evaluation.
	shared_ptr<Sampler> sampler;                           ///< The sampler.
	shared_ptr<MeasurementModel> initialMeasurementModel;  ///< The initial static measurement model.
	shared_ptr<AdaptiveMeasurementModel> measurementModel; ///< The adaptive measurement model.
	shared_ptr<PositionExtractor> extractor;               ///< The position extractor.
};

} /* namespace condensation */
#endif /* PARTIALLYADAPTIVECONDENSATIONTRACKER_HPP_ */

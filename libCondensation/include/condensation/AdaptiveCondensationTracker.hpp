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
 * Condensation tracker that adapts to the appearance of the tracked object over time.
 *
 * This condensation tracker uses a measurement model that adapts to the found object. Must be initialized before
 * being used. May be initialized several times, as it may need more than one frame with the corresponding object
 * position in order to work correctly.
 */
class AdaptiveCondensationTracker {
public:

	/**
	 * Constructs a new adaptive condensation tracker.
	 *
	 * @param[in] sampler The sampler.
	 * @param[in] measurementModel The adaptive measurement model.
	 * @param[in] extractor The position extractor.
	 * @param[in] initialCount The initial amount of particles.
	 */
	AdaptiveCondensationTracker(shared_ptr<Sampler> sampler,
			shared_ptr<AdaptiveMeasurementModel> measurementModel, shared_ptr<PositionExtractor> extractor, int initialCount);

	~AdaptiveCondensationTracker();

	/**
	 * Initializes this tracker at the given position. May need several subsequent initializations before
	 * being usable.
	 *
	 * @param[in] image The current image.
	 * @param[in] position The current position of the object that should be tracked.
	 * @return True if this tracker is usable, false otherwise.
	 */
	bool initialize(const Mat& image, const Rect& position);

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

private:

	int initialCount;          ///< The initial amount of particles.
	vector<Sample> samples;    ///< The current samples.
	vector<Sample> oldSamples; ///< The previous samples.
	optional<Sample> state;    ///< The estimated state.

	shared_ptr<VersionedImage> image;                      ///< The image used for evaluation.
	shared_ptr<Sampler> sampler;                           ///< The sampler.
	shared_ptr<AdaptiveMeasurementModel> measurementModel; ///< The adaptive measurement model.
	shared_ptr<PositionExtractor> extractor;               ///< The position extractor.
};

} /* namespace condensation */
#endif /* ADAPTIVECONDENSATIONTRACKER_HPP_ */

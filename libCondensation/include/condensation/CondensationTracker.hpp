/*
 * CondensationTracker.hpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef CONDENSATIONTRACKER_HPP_
#define CONDENSATIONTRACKER_HPP_

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
class PositionExtractor;

/**
 * Tracker of a single object in image/video streams based on the Condensation algorithm (aka Particle Filter).
 */
class CondensationTracker {
public:

	/**
	 * Constructs a new condensation tracker.
	 *
	 * @param[in] sampler The sampler.
	 * @param[in] measurementModel The measurement model.
	 * @param[in] extractor The position extractor.
	 */
	CondensationTracker(shared_ptr<Sampler> sampler, shared_ptr<MeasurementModel> measurementModel,
			shared_ptr<PositionExtractor> extractor);

	~CondensationTracker();

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

	vector<Sample> samples;    ///< The current samples.
	vector<Sample> oldSamples; ///< The previous samples.
	optional<Sample> state;    ///< The estimated state.

	shared_ptr<VersionedImage> image;              ///< The image used for evaluation.
	shared_ptr<Sampler> sampler;                   ///< The sampler.
	shared_ptr<MeasurementModel> measurementModel; ///< The measurement model.
	shared_ptr<PositionExtractor> extractor;       ///< The position extractor.
};

} /* namespace condensation */
#endif /* CONDENSATIONTRACKER_HPP_ */

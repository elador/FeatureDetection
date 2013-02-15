/*
 * CondensationTracker.h
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef CONDENSATIONTRACKER_H_
#define CONDENSATIONTRACKER_H_

#include "tracking/Sample.h"
#include "opencv2/core/core.hpp"
#include "boost/optional.hpp"
#include <memory>
#include <vector>

using cv::Mat;
using std::shared_ptr;
using boost::optional;
using std::vector;

namespace tracking {

class Rectangle;
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
	explicit CondensationTracker(shared_ptr<Sampler> sampler,
			shared_ptr<MeasurementModel> measurementModel, shared_ptr<PositionExtractor> extractor);

	~CondensationTracker();

	/**
	 * Processes the next image and returns the most probable object position.
	 *
	 * @param[in] image The next image.
	 * @return The bounding box around the most probable object position if there is an object.
	 */
	optional<Rectangle> process(const Mat& image);

	/**
	 * @return The current samples.
	 */
	inline const std::vector<Sample>& getSamples() const {
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

	optional<Sample> oldPosition; ///< The previous position.
	vector<double> offset;        ///< The movement of the tracked object's center of the previous time step.

	shared_ptr<Sampler> sampler;                   ///< The sampler.
	shared_ptr<MeasurementModel> measurementModel; ///< The measurement model.
	shared_ptr<PositionExtractor> extractor;       ///< The position extractor.
};

} /* namespace tracking */
#endif /* CONDENSATIONTRACKER_H_ */

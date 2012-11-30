/*
 * LearningCondensationTracker.h
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#ifndef LEARNINGCONDENSATIONTRACKER_H_
#define LEARNINGCONDENSATIONTRACKER_H_

#include "tracking/Sample.h"
#include "opencv2/core/core.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/optional.hpp"
#include <vector>

using boost::shared_ptr;
using boost::optional;
using cv::Mat;
using std::vector;

namespace tracking {

class Rectangle;
class Sampler;
class LearningMeasurementModel;
class PositionExtractor;
class LearningStrategy;

/**
 * Condensation tracker that learns the appearance of the tracked object over time.
 */
class LearningCondensationTracker {
public:

	/**
	 * Constructs a new learning condensation tracker.
	 *
	 * @param[in] sampler The sampler.
	 * @param[in] measurementModel The measurement model.
	 * @param[in] extractor The position extractor.
	 * @param[in] learningStrategy The learning strategy.
	 */
	explicit LearningCondensationTracker(shared_ptr<Sampler> sampler,
			shared_ptr<LearningMeasurementModel> measurementModel, shared_ptr<PositionExtractor> extractor,
			shared_ptr<LearningStrategy> learningStrategy);

	~LearningCondensationTracker();

	/**
	 * Processes the next image and returns the most probable object position.
	 *
	 * @param[in] image The next image.
	 * @return The bounding box around the most probable object position if there is an object.
	 */
	optional<Rectangle> process(Mat& image);

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
	 * @return True if learning is active, false otherwise.
	 */
	inline bool isLearningActive() {
		return learningActive;
	}

	/**
	 * @param[in] active Flag that indicates whether learning should be active.
	 */
	void setLearningActive(bool active);

private:

	vector<Sample> samples;    ///< The current samples.
	vector<Sample> oldSamples; ///< The previous samples.

	optional<Sample> oldPosition; ///< The previous position.
	vector<double> offset;        ///< The movement of the tracked object's center of the previous time step.
	bool learningActive;          ///< Flag that indicates whether learning is active.

	shared_ptr<Sampler> sampler;                           ///< The sampler.
	shared_ptr<LearningMeasurementModel> measurementModel; ///< The measurement model.
	shared_ptr<PositionExtractor> extractor;               ///< The position extractor.
	shared_ptr<LearningStrategy> learningStrategy;         ///< The learning strategy.
};

} /* namespace tracking */
#endif /* LEARNINGCONDENSATIONTRACKER_H_ */

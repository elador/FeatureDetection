/*
 * CondensationTracker.h
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef CONDENSATIONTRACKER_H_
#define CONDENSATIONTRACKER_H_

#include "tracking/Rectangle.h"
#include "tracking/Sample.h"

#include "boost/optional.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <vector>

class FdImage;

namespace tracking {

class ResamplingAlgorithm;
class TransitionModel;
class MeasurementModel;
class PositionExtractor;

/**
 * Tracker of objects in image/video streams based on the Condensation algorithm (aka Particle Filter).
 * For now, only a single object can be tracked.
 */
class CondensationTracker {
public:

	/**
	 * Constructs a new condensation tracker.
	 *
	 * @param[in] count The number of samples.
	 * @param[in] randomRate The percentage of samples that should be equally distributed.
	 * @param[in] resamplingAlgorithm The resampling algorithm.
	 * @param[in] transitionModel The transition model.
	 * @param[in] measurementModel The measurement model.
	 * @param[in] extractor The position extractor.
	 */
	explicit CondensationTracker(unsigned int count, double randomRate, ResamplingAlgorithm* resamplingAlgorithm,
			TransitionModel* transitionModel, MeasurementModel* measurementModel, PositionExtractor* extractor);
	virtual ~CondensationTracker();

	/**
	 * Processes the next image and returns the most probable object position.
	 *
	 * @param[in] image The next image.
	 * @return The bounding box around the most probable object position if there is an object.
	 */
	boost::optional<Rectangle> process(FdImage* image);

	/**
	 * @return The current samples.
	 */
	inline const std::vector<Sample>& getSamples() const {
		return samples;
	}

private:

	/**
	 * Determines whether a sample is valid for a certain image.
	 *
	 * @param[in] sample The sample.
	 * @param[in] image The image.
	 * @return True if the sample is valid, false otherwise.
	 */
	bool isValid(const Sample& sample, const FdImage* image);

	/**
	 * Randomly samples new valid values for a sample.
	 *
	 * @param[in,out] sample The sample.
	 * @param[in] image The image.
	 */
	void sampleValid(Sample& sample, const FdImage* image);

	unsigned int count;								///< The number of samples.
	double randomRate;								///< The percentage of samples that should be equally distributed.
	std::vector<Sample> samples;			///< The current samples.
	std::vector<Sample> oldSamples;		///< The previous samples.

	std::vector<double> offset;		///< The movement of the tracked object's center of the previous time step.

	ResamplingAlgorithm* resamplingAlgorithm;	///< The resampling algorithm.
	TransitionModel* transitionModel;					///< The transition model.
	MeasurementModel* measurementModel;				///< The measurement model.
	PositionExtractor* extractor;							///< The position extractor.

	boost::mt19937 generator; 					///< Random number generator.
	boost::uniform_int<> distribution;	///< Uniform integer distribution.
};

} /* namespace tracking */
#endif /* CONDENSATIONTRACKER_H_ */

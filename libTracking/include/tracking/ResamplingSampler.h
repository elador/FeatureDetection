/*
 * ResamplingSampler.h
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#ifndef RESAMPLINGSAMPLER_H_
#define RESAMPLINGSAMPLER_H_

#include "tracking/Sampler.h"
#include "boost/shared_ptr.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"

using boost::shared_ptr;

namespace tracking {

class ResamplingAlgorithm;
class TransitionModel;

/**
 * Creates new samples by resampling the previous ones and moving them according to a transition model.
 * Additionally, some samples are randomly sampled across the image.
 */
class ResamplingSampler : public Sampler {
public:

	/**
	 * Constructs a new resampling sampler.
	 *
	 * @param[in] count The number of samples.
	 * @param[in] randomRate The percentage of samples that should be equally distributed.
	 * @param[in] resamplingAlgorithm The resampling algorithm.
	 * @param[in] transitionModel The transition model.
	 */
	explicit ResamplingSampler(unsigned int count, double randomRate,
			shared_ptr<ResamplingAlgorithm> resamplingAlgorithm, shared_ptr<TransitionModel> transitionModel);

	~ResamplingSampler();

	void sample(const std::vector<Sample>& samples, const std::vector<double>& offset,
				const FdImage* image, std::vector<Sample>& newSamples);

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

	unsigned int count; ///< The number of samples.
	double randomRate;  ///< The percentage of samples that should be equally distributed.
	shared_ptr<ResamplingAlgorithm> resamplingAlgorithm; ///< The resampling algorithm.
	shared_ptr<TransitionModel> transitionModel;         ///< The transition model.

	boost::mt19937 generator;          ///< Random number generator.
	boost::uniform_int<> distribution; ///< Uniform integer distribution.
};

} /* namespace tracking */
#endif /* RESAMPLINGSAMPLER_H_ */

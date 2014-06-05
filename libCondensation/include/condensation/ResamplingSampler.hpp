/*
 * ResamplingSampler.hpp
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#ifndef RESAMPLINGSAMPLER_HPP_
#define RESAMPLINGSAMPLER_HPP_

#include "condensation/Sampler.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_real.hpp"
#include <memory>

namespace condensation {

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
	 * @param[in] minSize The minimum size of a sample.
	 * @param[in] maxSize The maximum size of a sample.
	 */
	ResamplingSampler(unsigned int count, double randomRate, std::shared_ptr<ResamplingAlgorithm> resamplingAlgorithm,
			std::shared_ptr<TransitionModel> transitionModel, int minSize, int maxSize);

	void init(const cv::Mat& image);

	void sample(const std::vector<std::shared_ptr<Sample>>& samples, std::vector<std::shared_ptr<Sample>>& newSamples,
			const cv::Mat& image, const std::shared_ptr<Sample> target);

	/**
	 * @return The number of samples.
	 */
	inline int getCount() {
		return count;
	}

	/**
	 * @param[in] The new number of samples.
	 */
	inline void setCount(unsigned int count) {
		this->count = count;
	}

	/**
	 * @return The percentage of samples that should be equally distributed.
	 */
	inline double getRandomRate() {
		return randomRate;
	}

	/**
	 * @param[in] The new percentage of samples that should be equally distributed.
	 */
	inline void setRandomRate(double randomRate) {
		this->randomRate = std::max(0.0, std::min(1.0, randomRate));
	}

private:

	/**
	 * Randomly samples new values for a sample.
	 *
	 * @param[in,out] sample The sample.
	 * @param[in] image The image.
	 */
	void sampleValues(Sample& sample, const cv::Mat& image);

	unsigned int count; ///< The number of samples.
	double randomRate;  ///< The percentage of samples that should be equally distributed.
	std::shared_ptr<ResamplingAlgorithm> resamplingAlgorithm; ///< The resampling algorithm.
	std::shared_ptr<TransitionModel> transitionModel;         ///< The transition model.

	int minSize; ///< The minimum size of a sample.
	int maxSize; ///< The maximum size of a sample.

	boost::mt19937 generator; ///< Random number generator.
	boost::uniform_int<> intDistribution;   ///< Uniform integer distribution.
	boost::uniform_real<> realDistribution; ///< Uniform real distribution.
};

} /* namespace condensation */
#endif /* RESAMPLINGSAMPLER_HPP_ */

/*
 * Sampler.h
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#ifndef SAMPLER_H_
#define SAMPLER_H_

#include "tracking/Sample.h"
#include "opencv2/core/core.hpp"
#include <vector>

namespace tracking {

/**
 * Creates new samples.
 */
class Sampler {
public:

	virtual ~Sampler() {}

	/**
	 * Creates new samples.
	 *
	 * @param[in] samples The vector containing the samples of the previous time step.
	 * @param[in] offset The movement of the tracked object's center of the previous time step.
	 * @param[in] image The new image.
	 * @param[in,out] newSamples The vector to insert the new samples into.
	 */
	virtual void sample(const std::vector<Sample>& samples, const std::vector<double>& offset,
			const cv::Mat& image, std::vector<Sample>& newSamples) = 0;
};

} /* namespace tracking */
#endif /* SAMPLER_H_ */

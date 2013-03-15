/*
 * Sampler.h
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#ifndef SAMPLER_H_
#define SAMPLER_H_

#include "opencv2/core/core.hpp"
#include <vector>

using cv::Mat;
using std::vector;

namespace condensation {

class Sample;

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
	virtual void sample(const vector<Sample>& samples, const vector<double>& offset, const Mat& image,
			vector<Sample>& newSamples) = 0;
};

} /* namespace condensation */
#endif /* SAMPLER_H_ */

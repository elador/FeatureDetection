/*
 * GridSampler.h
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#ifndef GRIDSAMPLER_H_
#define GRIDSAMPLER_H_

#include "condensation/Sampler.h"

namespace condensation {

/**
 * Creates new samples according to a grid (sliding-window like).
 */
class GridSampler : public Sampler {
public:

	/**
	 * Constructs a new grid sampler.
	 *
	 * @param[in] minSize The minimum size of a sample relative to the width or height of the image (whatever is smaller).
	 * @param[in] maxSize The maximum size of a sample relative to the width or height of the image (whatever is smaller).
	 * @param[in] sizeScale The scale factor of the size (beginning from the minimum size). Has to be greater than one.
	 * @param[in] stepSize The step size relative to the sample size.
	 */
	GridSampler(float minSize, float maxSize, float sizeScale, float stepSize);

	~GridSampler();

	void sample(const vector<Sample>& samples, const vector<double>& offset, const Mat& image,
			vector<Sample>& newSamples);

private:

	float minSize;   ///< The minimum size of a sample relative to the width or height of the image (whatever is smaller).
	float maxSize;   ///< The maximum size of a sample relative to the width or height of the image (whatever is smaller).
	float sizeScale; ///< The scale factor of the size (beginning from the minimum size).
	float stepSize;  ///< The step size relative to the sample size.
};

} /* namespace condensation */
#endif /* GRIDSAMPLER_H_ */

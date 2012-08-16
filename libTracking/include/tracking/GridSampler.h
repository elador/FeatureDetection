/*
 * GridSampler.h
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#ifndef GRIDSAMPLER_H_
#define GRIDSAMPLER_H_

#include "tracking/Sampler.h"

namespace tracking {

/**
 * Creates new samples according to a grid.
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
	explicit GridSampler(float minSize, float maxSize, float sizeScale, float stepSize);
	virtual ~GridSampler();

	void sample(const std::vector<Sample>& samples, const std::vector<double>& offset,
				const FdImage* image, std::vector<Sample>& newSamples);

private:

	float minSize;   ///< The minimum size of a sample relative to the width or height of the image (whatever is smaller).
	float maxSize;   ///< The maximum size of a sample relative to the width or height of the image (whatever is smaller).
	float sizeScale; ///< The scale factor of the size (beginning from the minimum size).
	float stepSize;  ///< The step size relative to the sample size.
};

} /* namespace tracking */
#endif /* GRIDSAMPLER_H_ */

/*
 * DensityPositionExtractor.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef DENSITYPOSITIONEXTRACTOR_H_
#define DENSITYPOSITIONEXTRACTOR_H_

#include "tracking/PositionExtractor.h"
#include "tracking/Sample.h"
#include <algorithm>
#include <cmath>

namespace tracking {

/**
 * Position extractor that uses kernel density estimation to deliver the most probable position.
 */
class DensityPositionExtractor : public PositionExtractor {
public:

	/**
	 * Constructs a new density position extractor.
	 *
	 * @param[in] bandwidth The kernel bandwidth used for density estimation.
	 */
	explicit DensityPositionExtractor(int bandwidth);
	virtual ~DensityPositionExtractor();

	boost::optional<Sample> extract(const std::vector<Sample>& samples);

protected:

	/**
	 * Computes the (weighted sample) density at a certain position.
	 *
	 * @param[in] samples The samples.
	 * @param[in] position The position.
	 * @return The density of the weighted samples at the given position.
	 */
	double computeDensity(const std::vector<Sample>& samples, const Sample& position);

	/**
	 * Computes the value of the three-dimensional kernel function of the difference between two samples.
	 *
	 * @param[in] s A sample.
	 * @param[in] t Another sample.
	 * @return The value of the kernel function.
	 */
	inline double getScaledKernelValue(const Sample& s, const Sample& t) {
		double x = invertedBandwidth * (s.getX() - t.getX());
		double y = invertedBandwidth * (s.getY() - t.getY());
		double size = invertedBandwidth * (s.getSize() - t.getSize());
		double norm = sqrt(x * x + y * y + size * size);
		return invertedBandwidthProduct * getKernelValue(norm);
	}

	/**
	 * Computes the value of the kernel function given the (scaled) parameter.
	 *
	 * @param[in] argument The (scaled) parameter.
	 * @return The value of the kernel function.
	 */
	inline double getKernelValue(double argument) {
		return std::max(0.0, 1.0 - argument);
	}

private:
	double invertedBandwidth;        ///< The inverted kernel bandwidth.
	double invertedBandwidthProduct; ///< The product of the inverted bandwidths of all three dimensions.
};

} /* namespace tracking */
#endif /* DENSITYPOSITIONEXTRACTOR_H_ */

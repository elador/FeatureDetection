/*
 * RbfKernel.hpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef RBFKERNEL_HPP_
#define RBFKERNEL_HPP_

#include "classification/Kernel.hpp"

namespace classification {

/**
 * A radial basis function kernel.
 */
class RbfKernel : public Kernel {
public:

	explicit RbfKernel();
	RbfKernel(double gamma);
	~RbfKernel();

	/**
	 * Computes the radial basis function kernel of ...
	 * TODO I will change this to cv::Mat, once it is tested and running!
	 * TODO Also, the kernel parameters have to be removed from compute(...)!
	 *
	 * @param[in] input The ...
	 * @return The result of the kernel computation
	 */
	inline double compute(unsigned char* data, unsigned char* support, int nDim) const {	// Write the implementation in the header to allow inlining?
		int dot = 0;
		int val2;
		int i;
		for (i = 0; i != nDim; ++i) {
			val2 = data[i] - support[i];
			dot += val2 * val2;
		}
		return (float)exp(-gamma*dot);
	};

private:
	double gamma; ///< Parameter of the radial basis function exp(-gamma*|u-v|^2). TODO CHECK WITH MR IMPLEMENTATION! REALLY? What about a) 0-255 / 0-1 scaling? And 1/gamma or gamma?
				 // When loading, we do gamma/255^2 or something because we convert [0..1] (in .mat) to [0..255]uchar in c++ lib
};

} /* namespace classification */
#endif /* RBFKERNEL_HPP_ */

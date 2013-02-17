/*
 * PolynomialKernel.hpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef POLYNOMIALKERNEL_HPP_
#define POLYNOMIALKERNEL_HPP_

#include "classification/Kernel.hpp"

namespace classification {

/**
 * A polynomial kernel funcion.
 */
class PolynomialKernel : public Kernel {
public:

	explicit PolynomialKernel();
	~PolynomialKernel();

	/**
	 * Computes the polynomial kernel of ...
	 *
	 * @param[in] input The ...
	 * @return The result of the kernel computation
	 */
	inline double compute(unsigned char* data, unsigned char* support, int nDim) const;	// Write the implementation in the header to allow inlining?
	// float kernel(unsigned char*, unsigned char*, int, float, float, int, int); // from DetSVM.h

	/*
float DetectorSVM::kernel(unsigned char* data, unsigned char* support, int nonLinType, float basisParam, float divisor, int polyPower, int nDim)
{
	int dot = 0;
	int val2;
	float out;
	float val;
	int i;
	switch (nonLinType) {
	case 1: // polynomial
		for (i = 0; i != nDim; ++i)
			dot += data[i] * support[i];
		out = (dot+basisParam)/divisor;
		val = out;
		for (i = 1; i < polyPower; i++)
			out *= val;
		return out;
	case 2: // RBF
		for (i = 0; i != nDim; ++i) {
			val2 = data[i] - support[i];
			dot += val2 * val2;
		}
		return (float)exp(-basisParam*dot);
	default: assert(0);
	}
	return 0;

}


	*/

private:
	
};

} /* namespace classification */
#endif /* POLYNOMIALKERNEL_HPP_ */

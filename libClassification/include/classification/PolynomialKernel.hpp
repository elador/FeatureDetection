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
#include "svm.h"

namespace classification {

/**
 * A polynomial kernel function of the form k(x, y) = (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
 */
class PolynomialKernel : public Kernel {
public:

	/**
	 * Constructs a new polynomial kernel.
	 *
	 * @param[in] alpha The slope &alpha of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	 * @param[in] constant The constant term c of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	 * @param[in] degree The degree d of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	 */
	explicit PolynomialKernel(double alpha, double constant = 0, int degree = 2) :
			alpha(alpha), constant(constant), degree(degree) {}

	~PolynomialKernel() {}

	// float kernel(unsigned char*, unsigned char*, int, float, float, int, int); // from DetSVM.h TODO remove this line
	double compute(const Mat& lhs, const Mat& rhs) const {
		return powi(alpha * lhs.dot(rhs) + constant, degree);
	}

	void setLibSvmParams(struct svm_parameter *param) const {
		param->kernel_type = POLY;
		param->degree = degree;
		param->coef0 = constant;
		param->gamma = alpha;
	}

	// TODO: as can be seen in the old implementation, the parameters were basisParam, divisor and polyPower
	// the computation of the new parameters from the old ones (if necessary, e.g. loading from old matlab files):
	// alpha = 1 / divisor
	// constant = basisParam / divisor
	// degree = polyPower
	/*TODO remove old implementation
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

}*/

private:
	
	/**
	 * Computes the n-th power of a floating point number, where n is an integer value.
	 *
	 * @param[in] base The base number.
	 * @param[in] exponent The integer exponent.
	 * @return The n-th power of the given number.
	 */
	double powi(double base, int exponent) const {
		double tmp = base, ret = 1.0;
		for (int t = exponent; t > 0; t /= 2) {
			if (t % 2 == 1)
				ret *= tmp;
			tmp = tmp * tmp;
		}
		return ret;
	}

 	double alpha;    ///< The slope &alpha of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	double constant; ///< The constant term c of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	int degree;      ///< The degree d of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
};

} /* namespace classification */
#endif /* POLYNOMIALKERNEL_HPP_ */

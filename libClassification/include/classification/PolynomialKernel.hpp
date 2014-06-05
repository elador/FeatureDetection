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
#include "classification/KernelVisitor.hpp"
#include <stdexcept>

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

	double compute(const cv::Mat& lhs, const cv::Mat& rhs) const {
		return powi(alpha * lhs.dot(rhs) + constant, degree);
	}

	void accept(KernelVisitor& visitor) const {
		visitor.visit(*this);
	}

	/**
	 * @return The slope &alpha of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	 */
	double getAlpha() const {
		return alpha;
	}

	/**
	 * @return The constant term c of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	 */
	double getConstant() const {
		return constant;
	}

	/**
	 * @return The degree d of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	 */
	int getDegree() const {
		return degree;
	}

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

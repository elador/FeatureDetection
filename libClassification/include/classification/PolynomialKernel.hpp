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
#include <stdexcept>

using std::invalid_argument;

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

	double compute(const Mat& lhs, const Mat& rhs) const {
		switch (lhs.type()) {
			case CV_8U: return powi(alpha * computeDotProduct_uchar(lhs, rhs) + constant, degree);
			case CV_32F: return powi(alpha * computeDotProduct_any<float>(lhs, rhs) + constant, degree);
		}
		throw invalid_argument("PolynomialKernel: arguments have to be of type CV_8U or CV_32F");
//		return powi(alpha * lhs.dot(rhs) + constant, degree);
	}

	void setLibSvmParams(struct svm_parameter *param) const {
		param->kernel_type = POLY;
		param->degree = degree;
		param->coef0 = constant;
		param->gamma = alpha;
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

	int computeDotProduct_uchar(const Mat& lhs, const Mat& rhs) const {
		const uchar* lvalues = lhs.ptr<uchar>(0);
		const uchar* rvalues = rhs.ptr<uchar>(0);
		int dot = 0;
		for (int i = 0; i < lhs.rows * lhs.cols; ++i)
			dot += lvalues[i] * rvalues[i];
		return dot;
	}

	template<class T>
	float computeDotProduct_any(const Mat& lhs, const Mat& rhs) const {
		const T* lvalues = lhs.ptr<T>(0);
		const T* rvalues = rhs.ptr<T>(0);
		float dot = 0;
		for (int i = 0; i < lhs.rows * lhs.cols; ++i)
			dot += lvalues[i] * rvalues[i];
		return dot;
	}

	double alpha;    ///< The slope &alpha of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	double constant; ///< The constant term c of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
	int degree;      ///< The degree d of the polynomial kernel function (&alpha; * x<sup>T</sup>y + c)<sup>d</sup>.
};

} /* namespace classification */
#endif /* POLYNOMIALKERNEL_HPP_ */

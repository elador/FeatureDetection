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
#include "svm.h"
#include <stdexcept>

using std::invalid_argument;

namespace classification {

/**
 * A radial basis function kernel of the form k(x, y) = exp(-&gamma; * |u - v|²).
 */
class RbfKernel : public Kernel {
public:

	/**
	 * Constructs a new RBF kernel.
	 *
	 * @param[in] gamma The parameter &gamma; of the radial basis function exp(-&gamma; * |u - v|²).
	 */
	explicit RbfKernel(double gamma) : gamma(gamma) {}

	~RbfKernel() {}

	double compute(const Mat& lhs, const Mat& rhs) const {
		if (lhs.channels() > 1 || rhs.channels() > 1)
			throw invalid_argument("RbfKernel: arguments have to have only one channel");
		if (lhs.flags != rhs.flags)
			throw invalid_argument("RbfKernel: arguments have to have the same type");
		if (lhs.rows * lhs.cols != rhs.rows * rhs.cols)
			throw invalid_argument("RbfKernel: arguments have to have the same length");
		return exp(-gamma * computeSumOfSquaredDifferences(lhs, rhs));
	}

	void setLibSvmParams(struct svm_parameter *param) const {
		param->kernel_type = RBF;
		param->gamma = gamma;
	}

private:

	/**
	 * Computes the sum of the squared differences of two vectors.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the squared differences.
	 */
	double computeSumOfSquaredDifferences(const Mat& lhs, const Mat& rhs) const {
		switch (lhs.type()) {
			case CV_8U: return computeSumOfSquaredDifferences_uchar(lhs, rhs);
			case CV_32S: return computeSumOfSquaredDifferences_any<int>(lhs, rhs);
			case CV_32F: return computeSumOfSquaredDifferences_any<float>(lhs, rhs);
		}
		throw invalid_argument("RbfKernel: arguments have to be of type CV_8U, CV_32S or CV_32F");
	}

	/**
	 * Computes the sum of the squared differences of two vectors of unsigned characters.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the squared differences.
	 */
	int computeSumOfSquaredDifferences_uchar(const Mat& lhs, const Mat& rhs) const {
		const uchar* lvalues = lhs.ptr<uchar>(0);
		const uchar* rvalues = rhs.ptr<uchar>(0);
		int sum = 0;
		for (int i = 0; i < lhs.rows * lhs.cols; ++i) {
			int diff = lvalues[i] - rvalues[i];
			sum += diff * diff;
		}
		return sum;
	}

	/**
	 * Computes the sum of the squared differences of two vectors of arbitrary value types.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the squared differences.
	 */
	template<class T>
	float computeSumOfSquaredDifferences_any(const Mat& lhs, const Mat& rhs) const {
		const T* lvalues = lhs.ptr<T>(0);
		const T* rvalues = rhs.ptr<T>(0);
		float sum = 0;
		for (int i = 0; i < lhs.rows * lhs.cols; ++i) {
			float diff = lvalues[i] - rvalues[i];
			sum += diff * diff;
		}
		return sum;
	}

	double gamma; ///< The parameter &gamma; of the radial basis function exp(-&gamma; * |u - v|²).
};

} /* namespace classification */
#endif /* RBFKERNEL_HPP_ */

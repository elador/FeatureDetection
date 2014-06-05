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
#include "classification/KernelVisitor.hpp"
#include <stdexcept>

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

	double compute(const cv::Mat& lhs, const cv::Mat& rhs) const {
		if (!lhs.isContinuous() || !rhs.isContinuous())
			throw std::invalid_argument("RbfKernel: arguments have to be continuous");
		if (lhs.flags != rhs.flags)
			throw std::invalid_argument("RbfKernel: arguments have to have the same type");
		if (lhs.total() != rhs.total())
			throw std::invalid_argument("RbfKernel: arguments have to have the same length");
		return exp(-gamma * computeSumOfSquaredDifferences(lhs, rhs));
	}

	void accept(KernelVisitor& visitor) const {
		visitor.visit(*this);
	}

	/**
	 * @return The parameter &gamma; of the radial basis function exp(-&gamma; * |u - v|²).
	 */
	double getGamma() const {
		return gamma;
	}

private:

	/**
	 * Computes the sum of the squared differences of two vectors.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the squared differences.
	 */
	double computeSumOfSquaredDifferences(const cv::Mat& lhs, const cv::Mat& rhs) const {
		switch (lhs.depth()) {
			case CV_8U: return computeSumOfSquaredDifferences_uchar(lhs, rhs);
			case CV_32S: return computeSumOfSquaredDifferences_any<int>(lhs, rhs);
			case CV_32F: return computeSumOfSquaredDifferences_any<float>(lhs, rhs);
		}
		throw std::invalid_argument("RbfKernel: arguments have to be of depth CV_8U, CV_32S or CV_32F");
	}

	/**
	 * Computes the sum of the squared differences of two vectors of unsigned characters.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the squared differences.
	 */
	int computeSumOfSquaredDifferences_uchar(const cv::Mat& lhs, const cv::Mat& rhs) const {
		const uchar* lvalues = lhs.ptr<uchar>();
		const uchar* rvalues = rhs.ptr<uchar>();
		int sum = 0;
		size_t size = lhs.total() * lhs.channels();
		for (size_t i = 0; i < size; ++i) {
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
	float computeSumOfSquaredDifferences_any(const cv::Mat& lhs, const cv::Mat& rhs) const {
		const T* lvalues = lhs.ptr<T>();
		const T* rvalues = rhs.ptr<T>();
		float sum = 0;
		size_t size = lhs.total() * lhs.channels();
		for (size_t i = 0; i < size; ++i) {
			float diff = lvalues[i] - rvalues[i];
			sum += diff * diff;
		}
		return sum;
	}

	double gamma; ///< The parameter &gamma; of the radial basis function exp(-&gamma; * |u - v|²).
};

} /* namespace classification */
#endif /* RBFKERNEL_HPP_ */

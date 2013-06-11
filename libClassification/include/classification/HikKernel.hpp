/*
 * HikKernel.hpp
 *
 *  Created on: 10.06.2013
 *      Author: poschmann
 */

#ifndef HIKKERNEL_HPP_
#define HIKKERNEL_HPP_

#include "classification/Kernel.hpp"
#include "svm.h"
#include <stdexcept>

using std::invalid_argument;

namespace classification {

/**
 * Histogram intersection kernel of the form k(x, y) = sum<sub>i</sub>(min(x<sub>i</sub>, y<sub>i</sub>)).
 */
class HikKernel : public Kernel {
public:

	/**
	 * Constructs a new histogram intersection kernel.
	 */
	HikKernel() {}

	~HikKernel() {}

	double compute(const Mat& lhs, const Mat& rhs) const {
		if (lhs.channels() > 1 || rhs.channels() > 1)
			throw invalid_argument("HikKernel: arguments have to have only one channel");
		if (lhs.flags != rhs.flags)
			throw invalid_argument("HikKernel: arguments have to have the same type");
		if (lhs.rows * lhs.cols != rhs.rows * rhs.cols)
			throw invalid_argument("HikKernel: arguments have to have the same length");
		return computeSumOfMinimums(lhs, rhs);
	}

	void setLibSvmParams(struct svm_parameter *param) const {
		param->kernel_type = HIK;
	}

private:

	/**
	 * Computes the sum over the minimums of two vectors.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the squared differences.
	 */
	double computeSumOfMinimums(const Mat& lhs, const Mat& rhs) const {
		switch (lhs.type()) {
			case CV_8U: return computeSumOfMinimums_uchar(lhs, rhs);
			case CV_32S: return computeSumOfMinimums_any<int>(lhs, rhs);
			case CV_32F: return computeSumOfMinimums_any<float>(lhs, rhs);
		}
		throw invalid_argument("HikKernel: arguments have to be of type CV_8U, CV_32S or CV_32F");
	}

	/**
	 * Computes the sum over the minimums of two vectors of unsigned characters.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the minimums.
	 */
	int computeSumOfMinimums_uchar(const Mat& lhs, const Mat& rhs) const {
		const uchar* lvalues = lhs.ptr<uchar>(0);
		const uchar* rvalues = rhs.ptr<uchar>(0);
		int sum = 0;
		for (int i = 0; i < lhs.rows * lhs.cols; ++i)
			sum += std::min(lvalues[i], rvalues[i]);
		return sum;
	}

	/**
	 * Computes the sum over the minimums of two vectors of arbitrary value types.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the minimums.
	 */
	template<class T>
	float computeSumOfMinimums_any(const Mat& lhs, const Mat& rhs) const {
		const T* lvalues = lhs.ptr<T>(0);
		const T* rvalues = rhs.ptr<T>(0);
		float sum = 0;
		for (int i = 0; i < lhs.rows * lhs.cols; ++i)
			sum += std::min(lvalues[i], rvalues[i]);
		return sum;
	}
};

} /* namespace classification */
#endif /* HIKKERNEL_HPP_ */

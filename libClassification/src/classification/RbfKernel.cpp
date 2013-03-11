/*
 * RbfKernel.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/RbfKernel.hpp"

namespace classification {

RbfKernel::RbfKernel(double gamma) : gamma(gamma) {}

RbfKernel::~RbfKernel() {}

double RbfKernel::computeSumOfSquaredDifferences(const Mat& lhs, const Mat& rhs) const {
//	if (lhs.type() != rhs.type() || lhs.size != rhs.size)
//		throw invalid_argument("RbfKernel: the given vectors have to be of the same type and size");
	// TODO wenn so io, dann wieder in hpp-datei verfrachten (inlining)
	Mat difference = lhs - rhs;
	return difference.dot(difference.t());
//	double sum = 0;
//	if (lhs.type() == CV_8U) {
//		const uint8_t* ls = lhs.ptr<uint8_t>(0);
//		const uint8_t* rs = rhs.ptr<uint8_t>(0);
//		for (int i = 0; i < lhs.cols; ++i, ++ls, ++rs) {
//			float d = static_cast<float>(*ls) - static_cast<float>(*rs);
//			sum += d * d;
//		}
//	} else if (lhs.type() == CV_32S) {
//		const int32_t* ls = lhs.ptr<int32_t>(0);
//		const int32_t* rs = rhs.ptr<int32_t>(0);
//		for (int i = 0; i < lhs.cols; ++i, ++ls, ++rs) {
//			double d = static_cast<double>(*ls) - static_cast<double>(*rs);
//			sum += d * d;
//		}
//	} else if (lhs.type() == CV_32F) {
//		const float* ls = lhs.ptr<float>(0);
//		const float* rs = rhs.ptr<float>(0);
//		for (int i = 0; i < lhs.cols; ++i, ++ls, ++rs) {
//			float d = *ls - *rs;
//			sum += d * d;
//		}
//	} else if (lhs.type() == CV_64F) {
//		const double* ls = lhs.ptr<double>(0);
//		const double* rs = rhs.ptr<double>(0);
//		for (int i = 0; i < lhs.cols; ++i, ++ls, ++rs) {
//			double d = *ls - *rs;
//			sum += d * d;
//		}
//	} else
//		throw invalid_argument("RbfKernel: the given vectors must be of type CV_8U, CV_32S, CV_32F or CV_64F");
//	return sum;
}

} /* namespace classification */

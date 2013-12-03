/*
 * KernelVisitor.hpp
 *
 *  Created on: 19.11.2013
 *      Author: poschmann
 */

#ifndef KERNELVISITOR_HPP_
#define KERNELVISITOR_HPP_

namespace classification {

class LinearKernel;
class PolynomialKernel;
class RbfKernel;
class HistogramIntersectionKernel;

/**
 * Kernel visitor.
 */
class KernelVisitor {
public:

	virtual ~KernelVisitor() {}

	/**
	 * Visits a linear kernel.
	 *
	 * @param[in] kernel The kernel.
	 */
	virtual void visit(const LinearKernel& kernel) = 0;

	/**
	 * Visits a polynomial kernel.
	 *
	 * @param[in] kernel The kernel.
	 */
	virtual void visit(const PolynomialKernel& kernel) = 0;

	/**
	 * Visits a RBF kernel.
	 *
	 * @param[in] kernel The kernel.
	 */
	virtual void visit(const RbfKernel& kernel) = 0;

	/**
	 * Visits a histogram intersection kernel.
	 *
	 * @param[in] kernel The kernel.
	 */
	virtual void visit(const HistogramIntersectionKernel& kernel) = 0;
};

} /* namespace classification */
#endif /* KERNELVISITOR_HPP_ */

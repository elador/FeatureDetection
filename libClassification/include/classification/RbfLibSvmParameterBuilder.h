/*
 * RbfLibSvmParameterBuilder.h
 *
 *  Created on: 17.10.2012
 *      Author: poschmann
 */

#ifndef RBFLIBSVMPARAMETERBUILDER_H_
#define RBFLIBSVMPARAMETERBUILDER_H_

#include "classification/LibSvmParameterBuilder.h"

namespace classification {

/**
 * Builder of parameters for libSVM training with an RBF kernel.
 */
class RbfLibSvmParameterBuilder : public LibSvmParameterBuilder {
public:

	/**
	 * Constructs a new RBF libSVM parameter builder.
	 *
	 * @param[in] gamma Parameter of the radial basis function exp(-gamma*|u-v|^2).
	 * @param[in] C The cost parameter of the SVM.
	 */
	explicit RbfLibSvmParameterBuilder(double gamma = 0.05, double C = 1);

	virtual ~RbfLibSvmParameterBuilder();

protected:

	struct svm_parameter *createBaseParameters();

private:

	double gamma; ///< Parameter of the radial basis function exp(-gamma*|u-v|^2).
};

} /* namespace classification */
#endif /* RBFLIBSVMPARAMETERBUILDER_H_ */

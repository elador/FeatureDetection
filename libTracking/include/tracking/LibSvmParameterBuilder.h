/*
 * LibSvmParameterBuilder.h
 *
 *  Created on: 17.10.2012
 *      Author: poschmann
 */

#ifndef LIBSVMPARAMETERBUILDER_H_
#define LIBSVMPARAMETERBUILDER_H_

struct svm_parameter;

namespace tracking {

/**
 * Builder of parameters for libSVM training.
 */
class LibSvmParameterBuilder {
public:

	/**
	 * Constructs a new libSVM parameter builder.
	 *
	 * @param[in] C The cost parameter of the SVM.
	 */
	LibSvmParameterBuilder(double C = 1);

	virtual ~LibSvmParameterBuilder() {}

	/**
	 * Creates the libSVM parameters. The memory has to be freed by calling svm_destroy_param.
	 *
	 * @param[in] positiveCount The amount of positive samples used for training.
	 * @param[in] negativeCount The amount of negative samples used for training.
	 * @return The parameters for libSVM training.
	 */
	struct svm_parameter *createParameters(unsigned int positiveCount, unsigned int negativeCount);

protected:

	/**
	 * Creates the libSVM parameters with some basic values that determine the used kernel and its properties.
	 * The properties that should be set (even if they are not used) are: C, kernel_type, gamma (for RBF and
	 * POLY kernel), degree, coef0 (both for POLY kernel).
	 *
	 * @return The incomplete parameters containing the kernel properties.
	 */
	virtual struct svm_parameter *createBaseParameters() = 0;

private:

	double C; ///< The cost parameter of the SVM.
};

} /* namespace tracking */
#endif /* LIBSVMPARAMETERBUILDER_H_ */

/*
 * PolyLibSvmParameterBuilder.h
 *
 *  Created on: 17.10.2012
 *      Author: poschmann
 */

#ifndef POLYLIBSVMPARAMETERBUILDER_H_
#define POLYLIBSVMPARAMETERBUILDER_H_

#include "classification/LibSvmParameterBuilder.h"

namespace classification {

/**
 * Builder of parameters for libSVM training with a polynomial kernel.
 */
class PolyLibSvmParameterBuilder : public LibSvmParameterBuilder {
public:

	/**
	 * Constructs a new polynomial libSVM parameter builder.
	 *
	 * @param[in] degree The degree of the polynomial.
	 * @param[in] coef The addition term of the polynomial function (gamma*u'*v + coef)^degree.
	 * @param[in] gamma The scale factor of the dot product of the polynomial function (gamma*u'*v + coef)^degree.
	 * @param[in] C The cost parameter of the SVM.
	 */
	explicit PolyLibSvmParameterBuilder(int degree = 3, double coef = 0, double gamma = 0.05, double C = 1);

	virtual ~PolyLibSvmParameterBuilder();

protected:

	struct svm_parameter *createBaseParameters();

private:

	int degree;   ///< The degree of the polynomial.
	double coef;  ///< The addition term of the polynomial function (gamma*u'*v + coef)^degree.
	double gamma; ///< The scale factor of the dot product of the polynomial function (gamma*u'*v + coef)^degree.
};

} /* namespace tracking */
#endif /* POLYLIBSVMPARAMETERBUILDER_H_ */

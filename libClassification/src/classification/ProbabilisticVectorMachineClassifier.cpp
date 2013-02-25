/*
 * ProbabilisticVectorMachineClassifier.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticVectorMachineClassifier.hpp"
#include "classification/VectorMachineClassifier.hpp"

namespace classification {

/*ProbabilisticVectorMachineClassifier::ProbabilisticVectorMachineClassifier( shared_ptr<VectorMachineClassifier> classifier ) : classifier(classifier)
{
	sigmoidParameters[0] = 0.0f;	// How to initialize this in an initializer list?
	sigmoidParameters[1] = 0.0f;
}*/


/*ProbabilisticVectorMachineClassifier::~ProbabilisticVectorMachineClassifier(void)
{
}*/


/*pair<bool, double> ProbabilisticVectorMachineClassifier::classify( const Mat& featureVector ) const
{
	pair<bool, double> res = classifier->classify(featureVector);
	// Do sigmoid stuff:
	res.second = 1.0f / (1.0f + exp(sigmoidParameters[0]*res.second + sigmoidParameters[1]));
	return res;

	/* TODO: In case of the WVM, we could again distinguish if we calculate the ("wrong") probability
	 *			of all patches or only the ones that pass the last stage (correct probability), and 
	 *			set all others to zero.
	//fp->certainty = 1.0f / (1.0f + exp(posterior_wrvm[0]*fout + posterior_wrvm[1]));
	// We ran till the REAL LAST filter (not just the numUsedFilters one):
	if(filter_level+1 == this->numLinFilters && fout >= this->hierarchicalThresholds[filter_level]) {
		certainty = 1.0f / (1.0f + exp(posterior_wrvm[0]*fout + posterior_wrvm[1]));
		return true;
	}
	// We didn't run up to the last filter (or only up to the numUsedFilters one)
	if(this->calculateProbabilityOfAllPatches==true) {
		certainty = 1.0f / (1.0f + exp(posterior_wrvm[0]*fout + posterior_wrvm[1]));
		return false;
	} else {
		certainty = 0.0f;
		return false;
	}
	*/
//}

} /* namespace classification */

/*
 * VectorMachineClassifier.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticVectorMachineClassifier.hpp"

namespace classification {

ProbabilisticVectorMachineClassifier::ProbabilisticVectorMachineClassifier(void)
{
	posterior_svm[0] = 0.0f;
	posterior_svm[1] = 0.0f;
}


ProbabilisticVectorMachineClassifier::~ProbabilisticVectorMachineClassifier(void)
{
}

} /* namespace classification */


/* From DetectorSVM:
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		std::cout << "[DetSVM] Unable to open the file (wrong format?):" << std::endl <<  thresholdsFilename << std::endl;
		return 1;
	} else {
		//read posterior_wrvm parameter for probabilistic WRVM output
		//TODO 2012: is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
		//TODO new2013: the posterior_svm is the same for all thresholds file, so the thresholds file is not needed for SVMs. The posterior_svm should go into the classifier.mat!
		pmxarray = matGetVariable(pmatfile, "posterior_svm");
		if (pmxarray == 0) {
			std::cout << "[DetSVM] WARNING: Unable to find the vector posterior_svm, disable prob. SVM output;" << std::endl;
			this->posterior_svm[0]=this->posterior_svm[1]=0.0f;
		} else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			if (dim[1] != 2) {
				std::cout << "[DetSVM] WARNING: Size of vector posterior_svm !=2, disable prob. SVM output;" << std::endl;
				this->posterior_svm[0]=this->posterior_svm[1]=0.0f;
			} else {
				this->posterior_svm[0]=(float)matdata[0]; this->posterior_svm[1]=(float)matdata[1];
			}
			mxDestroyArray(pmxarray);
		}
		matClose(pmatfile);
	}
	*/
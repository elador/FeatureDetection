/*
 * ProbabilisticWvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/VectorMachineClassifier.hpp"

#include "mat.h"
#include <iostream>

namespace classification {


void ProbabilisticWvmClassifier::load( const string classifierFilename, const string thresholdsFilename )
{
	// Load sigmoid stuff:
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		std::cout << "[DetWVM] : Unable to open the file (wrong format?):" << std::endl << thresholdsFilename << std::endl;
		exit(EXIT_FAILURE);
	} 

	//printf("fd_ReadDetector(): read posterior_svm parameter for probabilistic SVM output\n");
	//read posterior_wrvm parameter for probabilistic WRVM output
	//TODO is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
	pmxarray = matGetVariable(pmatfile, "posterior_wrvm");
	if (pmxarray == 0) {
		std::cout << "[DetWVM] WARNING: Unable to find the vector posterior_wrvm, disable prob. SVM output;" << std::endl;
		this->sigmoidParameters[0]=this->sigmoidParameters[1]=0.0f;
	} else {
		double* matdata = mxGetPr(pmxarray);
		const mwSize *dim = mxGetDimensions(pmxarray);
		if (dim[1] != 2) {
			std::cout << "[DetWVM] WARNING: Size of vector posterior_wrvm !=2, disable prob. WRVM output;" << std::endl;
			this->sigmoidParameters[0]=this->sigmoidParameters[1]=0.0f;
		} else {
			this->sigmoidParameters[0]=(float)matdata[0]; this->sigmoidParameters[1]=(float)matdata[1];
		}
		mxDestroyArray(pmxarray);
	}
	matClose(pmatfile);

	// Load the detector and thresholds:
	classifier->load(classifierFilename, thresholdsFilename);
}

} /* namespace classification */

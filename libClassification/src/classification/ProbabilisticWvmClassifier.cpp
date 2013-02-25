/*
 * ProbabilisticWvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"

#include "mat.h"
#include <iostream>

namespace classification {

ProbabilisticWvmClassifier::ProbabilisticWvmClassifier( shared_ptr<WvmClassifier> classifier )
{
	sigmoidParameters[0] = 0.0f;	// How to initialize this in an initializer list?
	sigmoidParameters[1] = 0.0f;
	this->classifier = classifier;	// TODO initializer list doesn't work? This is all not optimal......
	
}

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

pair<bool, double> ProbabilisticWvmClassifier::classify( const Mat& featureVector ) const
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
}

} /* namespace classification */

/*
 * ProbabilisticSvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "mat.h"
#include <iostream>

namespace classification {

ProbabilisticSvmClassifier::ProbabilisticSvmClassifier( shared_ptr<SvmClassifier> classifier )
{
	sigmoidParameters[0] = 0.0f;	// How to initialize this in an initializer list?
	sigmoidParameters[1] = 0.0f;
	classifier = classifier;	// TODO initializer list doesn't work? This is all not optimal......
}

void ProbabilisticSvmClassifier::load( const string classifierFilename, const string thresholdsFilename )
{
	// Load sigmoid stuff:
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		std::cout << "[DetSVM] Unable to open the file (wrong format?):" << std::endl <<  thresholdsFilename << std::endl;
		// THROW
	} else {
		//read posterior_wrvm parameter for probabilistic WRVM output
		//TODO 2012: is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
		//TODO new2013: the posterior_svm is the same for all thresholds file, so the thresholds file is not needed for SVMs. The posterior_svm should go into the classifier.mat!
		pmxarray = matGetVariable(pmatfile, "posterior_svm");
		if (pmxarray == 0) {
			std::cout << "[DetSVM] WARNING: Unable to find the vector posterior_svm, disable prob. SVM output;" << std::endl;
			this->sigmoidParameters[0]=this->sigmoidParameters[1]=0.0f;
		} else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			if (dim[1] != 2) {
				std::cout << "[DetSVM] WARNING: Size of vector posterior_svm !=2, disable prob. SVM output;" << std::endl;
				this->sigmoidParameters[0]=this->sigmoidParameters[1]=0.0f;
			} else {
				this->sigmoidParameters[0]=(float)matdata[0]; this->sigmoidParameters[1]=(float)matdata[1];
			}
			mxDestroyArray(pmxarray);
		}
		matClose(pmatfile);
	}

	// Load the detector and thresholds:
	classifier->load(classifierFilename, thresholdsFilename);
}

pair<bool, double> ProbabilisticSvmClassifier::classify( const Mat& featureVector ) const
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

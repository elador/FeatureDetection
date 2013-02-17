/*
 * SvmClassifier.cpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

#include "classification/SvmClassifier.hpp"
#include "Mat.h"
#include <iostream>

using std::make_pair;
using std::cout;

namespace classification {

SvmClassifier::SvmClassifier()
{
	numSV = 0;
	support = NULL;
	alpha = NULL;
	limitReliability = 0.0f;
}

SvmClassifier::~SvmClassifier()
{
	if(support != 0) {
		for(int i = 0; i < numSV; ++i)
			delete[] support[i];
	}
	delete[] support;
	delete[] alpha;
}

pair<bool, double> SvmClassifier::classify(const Mat& featureVector) const
{
	FdPatch* patch = new FdPatch;
	patch->data = new unsigned char[featureVector.getSize()];
	for (unsigned int i = 0; i < featureVector.getSize(); ++i)
		patch->data[i] = (unsigned char)(255 * featureVector[i] + 0.5f);
	// Check if UChar 1Chan ? Float? 
	// loop over featureVec, kernel, ...
	bool positive = svm->classify(featureVector);
	//start
	float res = -this->nonlin_threshold;	// TODO ASK MR Why minus???
	for (int i = 0; i != this->numSV; ++i) {
		res += this->alpha[i] * kernel(fp->data, this->support[i], this->nonLinType, this->basisParam, this->divisor, this->polyPower, this->filter_size_x*this->filter_size_y);
	}
	//fp->fout = res;
	std::pair<FoutMap::iterator, bool> fout_insert = fp->fout.insert(FoutMap::value_type(this->identifier, res));
	if(fout_insert.second == false) {
		std::cout << "[DetectorSVM] An element 'fout' already exists for this detector, you classified the same patch twice. This should never happen." << std::endl;
	}
	//fp->certainty = 1.0f / (1.0f + exp(posterior_svm[0]*res + posterior_svm[1]));
	std::pair<CertaintyMap::iterator, bool> certainty_insert = fp->certainty.insert(CertaintyMap::value_type(this->identifier, 1.0f / (1.0f + exp(posterior_svm[0]*res + posterior_svm[1]))));
	if(certainty_insert.second == false) {
		std::cout << "[DetectorSVM] An element 'certainty' already exists for this detector, you classified the same patch twice. This should never happen." << std::endl;
	}
	
	if (res >= this->limit_reliability) {
		return true;
	}
	return false;


	//end
	double probability = patch->certainty[wvm->getIdentifier()];

	delete[] patch->data;
	patch->data = NULL;
	delete patch;

	return make_pair(positive, probability);
}

void SvmClassifier::load(const std::string classifierFilename, const std::string thresholdsFilename, float limitReliability)
{
	std::cout << "[SvmClassifier] Loading " << classifierFilename << std::endl;

	this->limitReliability = limitReliability;
	
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	double *matdata;
	pmatfile = matOpen(fn_classifier, "r");
	if (pmatfile == NULL) {
		std::cout << "[DetSVM] Error opening file." << std::endl;
		exit(EXIT_FAILURE);
	}

	pmxarray = matGetVariable(pmatfile, "param_nonlin1");
	if (pmxarray == 0) {
		std::cout << "[DetSVM] Error: There is a no param_nonlin1 in the file." << std::endl;
		exit(EXIT_FAILURE);
	}
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] Reading param_nonlin1" << std::endl;
	}
	matdata = mxGetPr(pmxarray);
	this->nonlin_threshold = (float)matdata[0];
	this->nonLinType       = (int)matdata[1];
	this->basisParam       = (float)(matdata[2]/65025.0); // because the training image's graylevel values were divided by 255
	this->polyPower        = (int)matdata[3];
	this->divisor          = (float)matdata[4];
	mxDestroyArray(pmxarray);
		
	pmxarray = matGetVariable(pmatfile, "support_nonlin1");
	if (pmxarray == 0) {
		std::cout << "[DetSVM] Error: There is a nonlinear SVM in the file, but the matrix support_nonlin1 is lacking!" << std::endl;
		exit(EXIT_FAILURE);
	} 
	if (mxGetNumberOfDimensions(pmxarray) != 3) {
		std::cout << "[DetSVM] Error: The matrix support_nonlin1 in the file should have 3 dimensions." << std::endl;
		exit(EXIT_FAILURE);
	}
	const mwSize *dim = mxGetDimensions(pmxarray);
	this->numSV = (int)dim[2];
	matdata = mxGetPr(pmxarray);

	int is;
	this->filter_size_x = (int)dim[1];
	this->filter_size_y = (int)dim[0];

	// Alloc space for SV's and alphas (weights)
	this->support = new unsigned char* [this->numSV];
	int size = filter_size_x*filter_size_y;
	for (int i = 0; i < this->numSV; ++i) 
		this->support[i] = new unsigned char[size];
	this->alpha = new float [this->numSV];

	int k = 0;
	for (is = 0; is < (int)dim[2]; ++is)
		for (int x = 0; x < filter_size_x; ++x)	// row-first (ML-convention)
			for (int y = 0; y < filter_size_y; ++y)
				this->support[is][y*filter_size_x+x] = (unsigned char)(255.0*matdata[k++]);	 // because the training images grey level values were divided by 255;
	mxDestroyArray(pmxarray);

	pmxarray = matGetVariable(pmatfile, "weight_nonlin1");
	if (pmxarray == 0) {
		std::cout << "[DetSVM] Error: There is a nonlinear SVM in the file but the matrix threshold_nonlin is lacking." << std::endl;
		exit(EXIT_FAILURE);
	}
	matdata = mxGetPr(pmxarray);
	for (is = 0; is < this->numSV; ++is)
		this->alpha[is] = (float)matdata[is];
	mxDestroyArray(pmxarray);

	if (matClose(pmatfile) != 0) {
		std::cout << "[DetSVM] Error closing file." << std::endl;
	}

	
	pmatfile = matOpen(fn_threshold, "r");
	if (pmatfile == 0) {
		std::cout << "[DetSVM] Unable to open the file (wrong format?):" << std::endl <<  fn_threshold << std::endl;
		return 1;
	} else {
		//printf("fd_ReadDetector(): read posterior_svm parameter for probabilistic SVM output\n");
		//read posterior_wrvm parameter for probabilistic WRVM output
		//TODO is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
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
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] Done reading posterior_svm [" << posterior_svm[0] << ", " << posterior_svm[1] << "] from threshold file " << fn_threshold << std::endl;
	}
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=1) {
		std::cout << "[DetSVM] Done reading SVM!" << std::endl;
	}

	this->stretch_fac = 255.0f/(float)(filter_size_x*filter_size_y);	// HistEq64 initialization


}

} /* namespace classification */

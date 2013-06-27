/*
 * RvmClassifier.cpp
 *
 *  Created on: 14.06.2013
 *      Author: Patrik Huber
 */

#include "classification/RvmClassifier.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
#include "logging/LoggerFactory.hpp"
#include "mat.h"
#include <stdexcept>
#include <fstream>

using logging::Logger;
using logging::LoggerFactory;
using std::make_shared;
using std::make_pair;
using std::invalid_argument;
using std::runtime_error;

namespace classification {

RvmClassifier::RvmClassifier(shared_ptr<Kernel> kernel) :
		VectorMachineClassifier(kernel), supportVectors(), coefficients() {}

RvmClassifier::~RvmClassifier() {}

bool RvmClassifier::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

bool RvmClassifier::classify(double hyperplaneDistance) const {
	return hyperplaneDistance >= threshold;
}

double RvmClassifier::computeHyperplaneDistance(const Mat& featureVector) const {
	double distance = -bias;
	for (size_t i = 0; i < supportVectors.size(); ++i)
		distance += coefficients[i] * kernel->compute(featureVector, supportVectors[i]);
	return distance;
}

void RvmClassifier::setSvmParameters(vector<Mat> supportVectors, vector<float> coefficients, double bias) {
	this->supportVectors = supportVectors;
	this->coefficients = coefficients;
	this->bias = bias;
}

/*
shared_ptr<RvmClassifier> RvmClassifier::loadMatlab(const string& classifierFilename)
{
	shared_ptr<RvmClassifier> wvm = make_shared<RvmClassifier>();
	
	//Number filters to use
	wvm->numUsedFilters=280;	// Todo make dynamic (from script)

	//Grenze der Zuverlaesigkeit ab der Gesichter aufgenommen werden (Diffwert fr W-RSV's-Schwellen)
	// zB. +0.1 => weniger patches drueber(mehr rejected, langsamer),    dh. mehr fn(FRR), weniger fp(FAR)  und
	// zB. -0.1 => mehr patches drueber(mehr nicht rejected, schneller), dh. weniger fn(FRR), mehr fp(FAR)
	wvm->limitReliabilityFilter=0.0f;	// FD.limit_reliability_filter (for WVM) (without _filter, it's for the SVM)

	Logger logger = Loggers->getLogger("classification");
	logger.info("Loading WVM classifier from matlab file: " + classifierFilename);
	
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	double *matdata;
	pmatfile = matOpen(classifierFilename.c_str(), "r");
	if (pmatfile == NULL) {
		// TODO: Maybe we should also log all of those exceptions, instead of only throwing them?
		throw invalid_argument("WvmClassifier: Could not open the provided classifier filename: " + classifierFilename);
	}

	pmxarray = matGetVariable(pmatfile, "num_hk");
	if (pmxarray == 0) {
		throw runtime_error("WvmClassifier: There is a no num_hk in the classifier file.");
		// TODO (concerns the whole class): I think we leak memory here (all the MATFile and double pointers etc.)?
	}
	matdata = mxGetPr(pmxarray);
	int nfilter = (int)matdata[0];
	mxDestroyArray(pmxarray);
	logger.debug("Found " + lexical_cast<string>(nfilter) + " WVM filters.");

	pmxarray = matGetVariable(pmatfile, "wrvm");
	if (pmxarray != 0) { // read area
		logger.error("Found a structure 'wrvm' and trying to read the " + lexical_cast<string>(nfilter) + " non-linear filters support_hk* and weight_hk* at once. However, this is not implemented yet.");
		throw runtime_error("WvmClassifier: Reading all wvm filters at once using the structure 'wrvm' is not (yet) supported.");
		// Note: If we at one time need this, I'll implement and test it.
		/*
				//read dim. hxw of support_hk's 
				mxArray* msup=mxGetField(pmxarray, 0, "dim");
				if (msup == 0) {
					printf("\nfd_ReadDetector(): Unable to find the matrix \'wrvm.dim\'\n");
					return 0;
				}
				matdata = mxGetPr(msup);
				int w = (int)matdata[0];
				int h = (int)matdata[1];
				if(w==0 || h==0) {
					//error, stop
					return 0;
				}
				this->filter_size_x = w;
				this->filter_size_y = h;
				this->numSV = nfilter;


				numLinFilters = nfilter;
				linFilters = new float* [numLinFilters];
				for (i = 0; i < numLinFilters; ++i) 
					linFilters[i] = new float[nDim];
				lin_thresholds = new float [numLinFilters];
				hierarchicalThresholds = new float [numLinFilters];
				hkWeights = new float* [numLinFilters];
				for (i = 0; i < numLinFilters; ++i) 
					hkWeights[i] = new float[numLinFilters];


				//read matrix support_hk's w*h x nfilter
				//i.e. filter are vectorized to size(w*h) x 1 (row-wise in C format not col-wise as in matlab)
				msup=mxGetField(pmxarray,0,"support_hk");
				if (msup == 0) {
					sprintf(buff, "\nfd_ReadDetector(): Unable to find the matrix \'wrvm.support_hk\' in\n'%s' \n",args->classificator);
					fprintf(stderr,buff); PUT(buff);
					throw buff;	return 4;
				}
				if (mxGetNumberOfDimensions(msup) != 2) {
					fprintf(stderr, "\nThe matrix \'wrvm.support_hk\' in the file %s should have 2 dimensions, but has %d\7\n",args->classificator, mxGetNumberOfDimensions(msup));
					return 4;
				}
				const MYMWSIZE *dim = mxGetDimensions(msup);
				int size = (int)dim[0], n = (int)dim[1]; 

				if ( n!=nfilter || w*h!=size) {
					fprintf(stderr, "\nfd_ReadDetector(): The dimensions of matrix \'wrvm.support_hk\' should be (size=w*h=%d x nfilter=%d), but is %dx%d\7\n",w*h,nfilter,size,n);
					return 4;
				}
				//fprintf(stdout, "fd_ReadDetector(): dimensions of matrix \'wrvm.support_hk\' should be (size=w*h=%d x nfilter=%d), and is %dx%d\n",w*h,nfilter,size,n);
				
				matdata = mxGetPr(msup);
				for (int i = 0, j=0; i < nfilter; ++i) {
					for (k = 0; k < size; ++k)
						detector->linFilters[i][k] = 255.0f*(float)matdata[j++];	// because the training images grey level values were divided by 255;
				}
				
				//read matrix weight_hk (1,...,i,...,nfilter) x nfilter
				//only the first 0,...,i-1 are set, all other are set to 0 in the matrix weight_hk
				msup=mxGetField(pmxarray,0,"weight_hk");
				if (msup == 0) {
					sprintf(buff, "\nfd_ReadDetector(): Unable to find the matrix \'wrvm.weight_hk\' in\n'%s' \n",args->classificator);
					fprintf(stderr,buff); PUT(buff);
					throw buff;	return 4;
				}
				if (mxGetNumberOfDimensions(msup) != 2) {
					fprintf(stderr, "\nThe matrix \'wrvm.weight_hk\' in the file %s should have 2 dimensions, but has %d\7\n",args->classificator, mxGetNumberOfDimensions(msup));
					return 4;
				}
				dim = mxGetDimensions(msup);
				int n1 = (int)dim[0], n2 = (int)dim[1]; 

				if ( n1!=nfilter || n2!=nfilter) {
					fprintf(stderr, "\nfd_ReadDetector(): The dimensions of matrix \'wrvm.weight_hk\' should be (nfilter=%d x nfilter=%d), but is %dx%d\7\n",nfilter,nfilter,n1,n2);
					return 4;
				}
				//fprintf(stdout, "fd_ReadDetector(): dimensions of matrix \'wrvm.weight_hk\' should be (nfilter=%d x nfilter=%d), and is %dx%d\n",nfilter,nfilter,n1,n2);
				
				matdata = mxGetPr(msup);
				for (int i = 0, r=0; i < nfilter; ++i, r+=nfilter) {
					for (k = 0; k <= i; ++k)
						detector->hkWeights[i][k] = (float)matdata[r + k];	
				}

				//for (k = 0; k < 5; ++k)	fprintf(stdout, "%1.2f ",detector->hkWeights[4][k]);	
				//fprintf(stdout, "\n");	
	
				mxDestroyArray(pmxarray);
				printf("...done\n");
				*/


/*
	} else {	// read seq.
		logger.debug("No structure 'wrvm' found, thus reading the " + lexical_cast<string>(nfilter) + " non-linear filters support_hk* and weight_hk* sequentially (slower).");
		char str[100];
		sprintf(str, "support_hk%d", 1);
		pmxarray = matGetVariable(pmatfile, str);
		const mwSize *dim = mxGetDimensions(pmxarray);
		int h = (int)dim[0];
		int w = (int)dim[1];
		//assert(w && h);
		wvm->filter_size_x = w;	// TODO check if this is right with eg 24x16
		wvm->filter_size_y = h;

		wvm->numLinFilters = nfilter;
		wvm->linFilters = new float* [wvm->numLinFilters];
		for (int i = 0; i < wvm->numLinFilters; ++i)
			wvm->linFilters[i] = new float[w*h];
		
		//hierarchicalThresholds = new float [numLinFilters];
		wvm->hkWeights = new float* [wvm->numLinFilters];
		for (int i = 0; i < wvm->numLinFilters; ++i)
			wvm->hkWeights[i] = new float[wvm->numLinFilters];

		if (pmxarray == 0) {
			throw runtime_error("WvmClassifier: Unable to find the matrix 'support_hk1' in the classifier file.");
		}
		if (mxGetNumberOfDimensions(pmxarray) != 2) {
			throw runtime_error("WvmClassifier: The matrix 'filter1' in the classifier file should have 2 dimensions.");
		}
		mxDestroyArray(pmxarray);

		for (int i = 0; i < wvm->numLinFilters; i++) {

			sprintf(str, "support_hk%d", i+1);
			pmxarray = matGetVariable(pmatfile, str);
			if (pmxarray == 0) {
				throw runtime_error("WvmClassifier: Unable to find the matrix 'support_hk" + lexical_cast<string>(i+1) + "' in the classifier file.");
			}
			if (mxGetNumberOfDimensions(pmxarray) != 2) {
				throw runtime_error("WvmClassifier: The matrix 'filter" + lexical_cast<string>(i+1) + "' in the classifier file should have 2 dimensions.");
			}

			matdata = mxGetPr(pmxarray);
				
			int k = 0;
			for (int x = 0; x < wvm->filter_size_x; ++x)
				for (int y = 0; y < wvm->filter_size_y; ++y)
					wvm->linFilters[i][y*wvm->filter_size_x+x] = 255.0f*(float)matdata[k++];	// because the training images grey level values were divided by 255;
			mxDestroyArray(pmxarray);

			sprintf(str, "weight_hk%d", i+1);
			pmxarray = matGetVariable(pmatfile, str);
			if (pmxarray != 0) {
				const mwSize *dim = mxGetDimensions(pmxarray);
				if ((dim[1] != i+1) && (dim[0] != i+1)) {
					throw runtime_error("WvmClassifier: The matrix " + lexical_cast<string>(str) + " in the classifier file should have a dimensions 1x" + lexical_cast<string>(i+1) + " or " + lexical_cast<string>(i+1) + "x1");
				}
				matdata = mxGetPr(pmxarray);
				for (int j = 0; j <= i; ++j) {
					wvm->hkWeights[i][j] = (float)matdata[j];
				}
				mxDestroyArray(pmxarray);
			}
		}	// end for over numHKs
		logger.debug("Vectors and weights successfully read.");

	}// end else read vecs/weights sequentially
	
	pmxarray = matGetVariable(pmatfile, "param_nonlin1_rvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		wvm->bias = (float)matdata[0];
		int nonLinType       = (int)matdata[1];
		wvm->basisParam       = (float)(matdata[2]/65025.0); // because the training images gray level values were divided by 255
		int polyPower        = (int)matdata[3];
		float divisor          = (float)matdata[4];
		mxDestroyArray(pmxarray);
	} else {
		pmxarray = matGetVariable(pmatfile, "param_nonlin1");
		if (pmxarray != 0) {
			matdata = mxGetPr(pmxarray);
			wvm->bias = (float)matdata[0];
			int nonLinType       = (int)matdata[1];
			wvm->basisParam       = (float)(matdata[2]/65025.0); // because the training images gray level values were divided by 255
			int polyPower        = (int)matdata[3];
			float divisor          = (float)matdata[4];
			mxDestroyArray(pmxarray);
		}
	}
	wvm->lin_thresholds = new float [wvm->numLinFilters];
	for (int i = 0; i < wvm->numLinFilters; ++i) {			//wrvm_out=treshSVM+sum(beta*kernel)
		wvm->lin_thresholds[i] = (float)wvm->bias;
	}

	// number of filters per level (eg 14)
	pmxarray = matGetVariable(pmatfile, "num_hk_wvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		assert(matdata != 0);	// TODO REMOVE
		wvm->numFiltersPerLevel = (int)matdata[0];
		mxDestroyArray(pmxarray);
	} else {
		throw runtime_error("WvmClassifier: Variable 'num_hk_wvm' not found in classifier file.");
	}
	// number of levels with filters (eg 20)
	pmxarray = matGetVariable(pmatfile, "num_lev_wvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		assert(matdata != 0);
		wvm->numLevels = (int)matdata[0];
		mxDestroyArray(pmxarray);
	} else {
		throw runtime_error("WvmClassifier: Variable 'num_lev_wvm' not found in classifier file.");
	}

	//read rectangles in area
	logger.debug("Reading rectangles in area...");
	pmxarray = matGetVariable(pmatfile, "area");
	if (pmxarray != 0 && mxIsStruct(pmxarray)) {
		int r,v,hrsv,w,h;
		int *cntrec;
		double *d;
		TRec *rec;
		const char valstr[]="val_u";
		const char cntrecstr[]="cntrec_u";

		w=wvm->filter_size_x; // again?
		h=wvm->filter_size_y;

		const mwSize *dim=mxGetDimensions(pmxarray);
		int nHK=(int)dim[1];
		if (wvm->numLinFilters!=nHK){
			throw runtime_error("WvmClassifier: Variable 'area' in the classifier file has wrong dimensions:" + lexical_cast<string>(nHK) + "(==" + lexical_cast<string>(wvm->numLinFilters) + ")");
		}
		if ((wvm->numFiltersPerLevel*wvm->numLevels)!=nHK){
			throw runtime_error("WvmClassifier: Variable 'area' in the classifier file has wrong dimensions:" + lexical_cast<string>(nHK) +  "(==" + lexical_cast<string>(wvm->numLinFilters) + ")");
		}

		wvm->area = new Area*[nHK];

		int cntval;
		for (hrsv=0;hrsv<nHK;hrsv++)  {

			mxArray* mval=mxGetField(pmxarray,hrsv,valstr);
 			if (mval == NULL ) {
				throw runtime_error("WvmClassifier: '" + lexical_cast<string>(valstr) + "' not found (WVM: 'val_u', else: 'val', right *.mat/kernel?)");
			}
			dim=mxGetDimensions(mval);
			cntval=(int)dim[1];

			cntrec = new int[cntval];
			mxArray* mcntrec=mxGetField(pmxarray,hrsv,cntrecstr);
			d = mxGetPr(mcntrec);
			for (v=0;v<cntval;v++) 
				cntrec[v]=(int)d[v];
			wvm->area[hrsv] = new Area(cntval,cntrec);

			d = mxGetPr(mval);
			for (v=0;v<cntval;v++) 
				wvm->area[hrsv]->val[v]=d[v]*255.0F; // because the training images grey level values were divided by 255;

			mxArray* mrec=mxGetField(pmxarray,hrsv,"crec");
			for (v=0;v<cntval;v++) 
				for (r=0;r<wvm->area[hrsv]->cntrec[v];r++) {
					mxArray* mk;
					rec=&wvm->area[hrsv]->rec[v][r];

					mk=mxGetField(mrec,r*cntval+v,"x1");
					d = mxGetPr(mk); rec->x1=(int)d[0];

					mk=mxGetField(mrec,r*cntval+v,"y1");
					d = mxGetPr(mk); rec->y1=(int)d[0];

					mk=mxGetField(mrec,r*cntval+v,"x2");
					d = mxGetPr(mk); rec->x2=(int)d[0];

					mk=mxGetField(mrec,r*cntval+v,"y2");
					d = mxGetPr(mk); rec->y2=(int)d[0];

					rec->uull=(rec->y1-1)*w + rec->x1-1;
					rec->uur= (rec->y1-1)*w + rec->x2;
					rec->dll= (rec->y2)*w   + rec->x1-1;
					rec->dr=  (rec->y2)*w   + rec->x2;
				}
	
			delete [] cntrec;

			//char name[255];
			//if (args->moreOutput) { 
			//sprintf(name,"%d",hrsv); this->area[hrsv]->dump(name);
			//}

		}
		mxDestroyArray(pmxarray);

	} else {
		throw runtime_error("WvmClassifier: 'area' not found (right *.mat/kernel?)");
	}

	//read convolution of the appr. rsv (pp) in	mat file
	pmxarray = matGetVariable(pmatfile, "app_rsv_convol");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		const mwSize *dim = mxGetDimensions(pmxarray);
		int nHK=(int)dim[1];
		if (wvm->numLinFilters!=nHK){
			throw runtime_error("WvmClassifier: 'app_rsv_convol' not right dim:" + lexical_cast<string>(nHK) + " (==" + lexical_cast<string>(wvm->numLinFilters) + ")");
		}
		wvm->app_rsv_convol = new double[nHK];
		for (int hrsv=0; hrsv<nHK; ++hrsv)
			wvm->app_rsv_convol[hrsv]=matdata[hrsv]*65025.0; // because the training images grey level values were divided by 255;
		mxDestroyArray(pmxarray);
	} else {
		throw runtime_error("WvmClassifier: 'app_rsv_convol' not found.");
	}

	if (matClose(pmatfile) != 0) {
		logger.warn("WvmClassifier: Could not close file " + classifierFilename);
		// TODO What is this? An error? Info? Throw an exception?
	}
	logger.info("WVM successfully read.");


	//printf("fd_ReadDetector(): making the hierarchical thresholds\n");
	// making the hierarchical thresholds
	//MATFile *mxtFile = matOpen(args->threshold, "r");
	logger.info("Loading WVM thresholds from matlab file: " + thresholdsFilename);
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		throw runtime_error("WvmClassifier: Unable to open the thresholds file (wrong format?):" + thresholdsFilename);
	} else {
		pmxarray = matGetVariable(pmatfile, "hierar_thresh");
		if (pmxarray == 0) {
			throw runtime_error("WvmClassifier: Unable to find the matrix hierar_thresh in the thresholds file.");
		} else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			for (int o=0; o<(int)dim[1]; ++o) {
				//TPairIf p(o+1, (float)matdata[o]);
				//std::pair<int, float> p(o+1, (float)matdata[o]); // = std::make_pair<int, float>
				//this->hierarchicalThresholdsFromFile.push_back(p);
				wvm->hierarchicalThresholdsFromFile.push_back((float)matdata[o]);
			}
			mxDestroyArray(pmxarray);
		}
			
		matClose(pmatfile);
	}

	/*int i;
	for (i = 0; i < this->numLinFilters; ++i) {
		this->hierarchicalThresholds[i] = 0;
	}
	for (i = 0; i < this->hierarchicalThresholdsFromFile.size(); ++i) {
		if (this->hierarchicalThresholdsFromFile[i].first <= this->numLinFilters)
			this->hierarchicalThresholds[this->hierarchicalThresholdsFromFile[i].first-1] = this->hierarchicalThresholdsFromFile[i].second;
	}
	//Diffwert fuer W-RSV's-Schwellen 
	if (this->limitReliabilityFilter!=0.0)
		for (i = 0; i < this->numLinFilters; ++i) this->hierarchicalThresholds[i]+=this->limitReliabilityFilter;
	*/
/*	if(wvm->hierarchicalThresholdsFromFile.size() != wvm->numLinFilters) {
		throw runtime_error("WvmClassifier: Something seems to be wrong, hierarchicalThresholdsFromFile.size() != numLinFilters; " + lexical_cast<string>(wvm->hierarchicalThresholdsFromFile.size()) + "!=" + lexical_cast<string>(wvm->numLinFilters));
	}
	wvm->setLimitReliabilityFilter(wvm->limitReliabilityFilter);	// This initializes the vector hierarchicalThresholds

	//for (i = 0; i < this->numLinFilters; ++i) printf("b%d=%g ",i+1,this->hierarchicalThresholds[i]);
	//printf("\n");
	logger.info("WVM thresholds successfully read.");

	wvm->filter_output = new float[wvm->numLinFilters];
	wvm->u_kernel_eval = new float[wvm->numLinFilters];

	wvm->setNumUsedFilters(wvm->numUsedFilters);	// Makes sure that we don't use more filters than the loaded WVM has, and if zero, set to numLinFilters.

	return wvm;
}
*/

shared_ptr<RvmClassifier> RvmClassifier::loadText(const string& classifierFilename)
{
	Logger logger = Loggers->getLogger("classification");
	logger.info("Loading SVM classifier from text file: " + classifierFilename);

	std::ifstream file(classifierFilename.c_str());
	if (!file.is_open())
		throw runtime_error("SvmClassifier: Invalid classifier file");

	string line;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");

	// read kernel parameters
	shared_ptr<Kernel> kernel;
	std::istringstream lineStream(line);
	if (lineStream.good() && !lineStream.fail()) {
		string kernelType;
		lineStream >> kernelType;
		if (kernelType != "FullPolynomial")
			throw runtime_error("SvmClassifier: Invalid kernel type: " + kernelType);
		int degree;
		double constant, scale;
		lineStream >> degree >> constant >> scale;
		kernel.reset(new PolynomialKernel(scale, constant, degree));
	}

	shared_ptr<RvmClassifier> svm = make_shared<RvmClassifier>(kernel);

	int svCount;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	std::sscanf(line.c_str(), "Number of SV : %d", &svCount);

	int dimensionCount;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	std::sscanf(line.c_str(), "Dim of SV : %d", &dimensionCount);

	float bias;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	std::sscanf(line.c_str(), "B0 : %f", &bias);
	svm->bias = bias;

	// coefficients
	svm->coefficients.resize(svCount);
	for (int i = 0; i < svCount; ++i) {
		float alpha;
		int index;
		if (!std::getline(file, line))
			throw runtime_error("SvmClassifier: Invalid classifier file");
		std::sscanf(line.c_str(), "alphas[%d]=%f", &index, &alpha);
		svm->coefficients[index] = alpha;
	}

	// read line containing "Support vectors: "
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	// read support vectors
	svm->supportVectors.reserve(svCount);
	for (int i = 0; i < svCount; ++i) {
		Mat vector(1, dimensionCount, CV_32F);
		if (!std::getline(file, line))
			throw runtime_error("SvmClassifier: Invalid classifier file");
		std::istringstream lineStream(line);
		if (!lineStream.good() || lineStream.fail())
			throw runtime_error("SvmClassifier: Invalid classifier file");
		float* values = vector.ptr<float>(0);
		for (int j = 0; j < dimensionCount; ++j)
			lineStream >> values[j];
		svm->supportVectors.push_back(vector);
	}

	logger.info("SVM successfully read.");

	return svm;
}

} /* namespace classification */

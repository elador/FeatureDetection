/*
 * WvmClassifier.cpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann & huber
 */

#include "classification/WvmClassifier.hpp"
#include "classification/IImg.hpp"
#include "mat.h"
#include <iostream>

using std::make_shared;
using std::make_pair;
using std::cout;

namespace classification {

WvmClassifier::WvmClassifier()
{
	linFilters			= NULL;
	lin_thresholds		= NULL;
	numLinFilters		= 0;
	numUsedFilters		= 0;
	hkWeights			= NULL;

	area = NULL;
	app_rsv_convol = NULL;

	filter_output = NULL;
	u_kernel_eval = NULL;

	limitReliabilityFilter = 0.0f;

	basisParam = 0.0f;

}

WvmClassifier::~WvmClassifier()
{
	if (linFilters != NULL) {
		for (int i = 0; i < numLinFilters; ++i)
			delete [] linFilters[i];
	}
	delete [] linFilters;
	if (hkWeights != NULL)
		for (int i = 0; i < numLinFilters; ++i) delete [] hkWeights[i];
	delete [] hkWeights;
	delete [] lin_thresholds;
	//delete [] hierarchicalThresholds;

	if (area!=NULL)	 {
		for (int i=0;i<numLinFilters;i++)
			if (area[i]!=NULL) 
				delete area[i];
		delete[] area;
	}
	if (app_rsv_convol!=NULL) delete [] app_rsv_convol;

	if (filter_output!=NULL) delete [] filter_output;
	if (u_kernel_eval!=NULL) delete [] u_kernel_eval;
}

bool WvmClassifier::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

bool WvmClassifier::classify(pair<int, double> levelAndDistance) const {
	// TODO the following todo was moved here from the end of the getHyperplaneDistance function (was in classify before)
	// TODO: filter statistics, nDropedOutAsNonFace[filter_level]++;
	// We ran till the REAL LAST filter (not just the numUsedFilters one), save the certainty
	int filterLevel = levelAndDistance.first;
	double fout = levelAndDistance.second;
	return filterLevel + 1 == this->numLinFilters && fout >= this->hierarchicalThresholds[filterLevel];
}

pair<int, double> WvmClassifier::computeHyperplaneDistance(const Mat& featureVector) const {
	unsigned int featureVectorLength = featureVector.rows * featureVector.cols;
	unsigned char* data = new unsigned char[featureVectorLength];

	int w = featureVector.cols;
	int h = featureVector.rows;
	for (int i=0; i<featureVector.rows; i++)
	{
		for (int j=0; j<featureVector.cols; j++)
		{
			data[i*w+j] = featureVector.at<uchar>(i, j); // (y, x) !!! i=row, j=column (matrix)
		}
	}

	//for (unsigned int i = 0; i < featureVectorLength; ++i)
	//data[i] = (featureVector.at<unsigned char>(i));
	//data[i] = (unsigned char)(255 * featureVector.at<float>(i) + 0.5f);
	// TODO Check if UChar 1Chan ? Float? 


	// STAAAAAAAAAAAART

	// Check if the patch has already been classified by this detector! If yes, don't do it again.
	// OK. We can't do this here! Because we do not know/save "filter_level". So we don't know when
	// the patch dropped out. Only the fout-value is not sufficient. So we can't know if we should
	// return true or false.
	// Possible solution: Store the filter_level somewhere.

	// So: The fout value is not already computed. Go ahead.

	// patch II of fp already calc'ed?
	
	IImg* iimg_x = new IImg(this->filter_size_x, this->filter_size_y, 8);
	iimg_x->calIImgPatch(data, false);
	IImg* iimg_xx = new IImg(this->filter_size_x, this->filter_size_y, 8);
	iimg_xx->calIImgPatch(data, true);

	for (int n=0;n<this->numFiltersPerLevel;n++) {
		u_kernel_eval[n]=0.0f;
	}
	int filter_level=-1;
	float fout = 0.0;
	do {
		filter_level++;
		fout = this->linEvalWvmHisteq64(filter_level, (filter_level%this->numFiltersPerLevel), filter_output, u_kernel_eval, iimg_x, iimg_xx);
		//} while (fout >= this->hierarchicalThresholds[filter_level] && filter_level+1 < this->numLinFilters); //280
	} while (fout >= this->hierarchicalThresholds[filter_level] && filter_level+1 < this->numUsedFilters); //280

	// fout = final result now!
	delete iimg_x;
	delete iimg_xx;

	// EEEEEEEEEEEEEEND

	delete[] data;
	data = NULL;
	return make_pair(filter_level, fout);
}

shared_ptr<WvmClassifier> WvmClassifier::load(const string& classifierFilename, const string& thresholdsFilename)
{
	shared_ptr<WvmClassifier> wvm = make_shared<WvmClassifier>();
	
	//Number filters to use
	wvm->numUsedFilters=280;	// Todo make dynamic (from script)

	//Grenze der Zuverlaesigkeit ab der Gesichter aufgenommen werden (Diffwert fr W-RSV's-Schwellen)
	// zB. +0.1 => weniger patches dr�ber(mehr rejected, langsamer),    dh. mehr fn(FRR), weniger fp(FAR)  und
	// zB. -0.1 => mehr patches dr�ber(mehr nicht rejected, schneller), dh. weniger fn(FRR), mehr fp(FAR)
	wvm->limitReliabilityFilter=0.0f;	// FD.limit_reliability_filter (for WVM) (without _filter, it's for the SVM)

	std::cout << "[WvmClassifier] Loading " << classifierFilename << std::endl;
	
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	double *matdata;
	pmatfile = matOpen(classifierFilename.c_str(), "r");
	if (pmatfile == NULL) {
		std::cout << "[WvmClassifier] Error opening file." << std::endl;
		exit(EXIT_FAILURE);
	}

	pmxarray = matGetVariable(pmatfile, "num_hk");
	if (pmxarray == 0) {
		std::cout << "[WvmClassifier] Error: There is a no num_hk in the file." << std::endl;
		exit(EXIT_FAILURE);
	}
	matdata = mxGetPr(pmxarray);
	int nfilter = (int)matdata[0];
	mxDestroyArray(pmxarray);
	std::cout << "[WvmClassifier] Found " << nfilter << " WVM filters" << std::endl;

	pmxarray = matGetVariable(pmatfile, "wrvm");
	if (pmxarray != 0) { // read area
		std::cout << "[WvmClassifier] Found structur 'wrvm', reading the TODO 280 non linear filters support_hk* and weight_hk* at once" << std::endl;
		std::cout << "[WvmClassifier] Stopping, because this is not yet tested!" << std::endl;
		exit(EXIT_FAILURE);
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



	} else {	// read seq.
		std::cout << "[WvmClassifier] Unable to find structur 'wrvm', reading the " << nfilter << " non linear filters support_hk* and weight_hk* sequentially (slower)" << std::endl;
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
			std::cout << "[WvmClassifier]  Unable to find the matrix \'support_hk1\'" << std::endl;
			exit(EXIT_FAILURE);
		}
		if (mxGetNumberOfDimensions(pmxarray) != 2) {
			std::cout << "[WvmClassifier] The matrix \'filter1\' in the file should have 2 dimensions" << std::endl;
			exit(EXIT_FAILURE);
		}
		mxDestroyArray(pmxarray);

		for (int i = 0; i < wvm->numLinFilters; i++) {

			sprintf(str, "support_hk%d", i+1);
			pmxarray = matGetVariable(pmatfile, str);
			if (pmxarray == 0) {
				std::cout << "[WvmClassifier] Unable to find the matrix \'support_hk" << i+1 << "\'" << std::endl;
				exit(EXIT_FAILURE);
			}
			if (mxGetNumberOfDimensions(pmxarray) != 2) {
				std::cout << "[WvmClassifier] The matrix \'filter" << i+1 << "\' in the file should have 2 dimensions" << std::endl;
				exit(EXIT_FAILURE);
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
					std::cout << "[WvmClassifier] The matrix " << str << " in the file should have a dimensions 1x" << i+1 << " or " << i+1 << "x1" << std::endl;
					exit(EXIT_FAILURE);
				}
				matdata = mxGetPr(pmxarray);
				for (int j = 0; j <= i; ++j) {
					wvm->hkWeights[i][j] = (float)matdata[j];
				}
				mxDestroyArray(pmxarray);
			}
		}	// end for over numHKs
		std::cout << "[WvmClassifier] Done" << std::endl;

	}// end else read vecs/weights sequentially
	
	pmxarray = matGetVariable(pmatfile, "param_nonlin1_rvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		wvm->nonlinThreshold = (float)matdata[0];
		int nonLinType       = (int)matdata[1];
		wvm->basisParam       = (float)(matdata[2]/65025.0); // because the training images grey level values were divided by 255
		int polyPower        = (int)matdata[3];
		float divisor          = (float)matdata[4];
		mxDestroyArray(pmxarray);
	} else {
		pmxarray = matGetVariable(pmatfile, "param_nonlin1");
		if (pmxarray != 0) {
			matdata = mxGetPr(pmxarray);
			wvm->nonlinThreshold = (float)matdata[0];
			int nonLinType       = (int)matdata[1];
			wvm->basisParam       = (float)(matdata[2]/65025.0); // because the training images grey level values were divided by 255
			int polyPower        = (int)matdata[3];
			float divisor          = (float)matdata[4];
			mxDestroyArray(pmxarray);
		}
	}
	wvm->lin_thresholds = new float [wvm->numLinFilters];
	for (int i = 0; i < wvm->numLinFilters; ++i) {			//wrvm_out=treshSVM+sum(beta*kernel)
		wvm->lin_thresholds[i] = (float)wvm->nonlinThreshold;
	}

	// number of filters per level (eg 14)
	pmxarray = matGetVariable(pmatfile, "num_hk_wvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		assert(matdata != 0);	// TODO REMOVE
		wvm->numFiltersPerLevel = (int)matdata[0];
		mxDestroyArray(pmxarray);
	} else {
		std::cout << "[WvmClassifier] : 'num_hk_wvm' not found in:" << std::endl;
		exit(EXIT_FAILURE);
	}
	// number of levels with filters (eg 20)
	pmxarray = matGetVariable(pmatfile, "num_lev_wvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		assert(matdata != 0);
		wvm->numLevels = (int)matdata[0];
		mxDestroyArray(pmxarray);
	} else {
		std::cout << "[WvmClassifier] : 'num_lev_wvm' not found in:" << std::endl;
		exit(EXIT_FAILURE);
	}

  
	//read recangles in area
	std::cout << "[WvmClassifier] Reading rectangles in area..." << std::endl;

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
			std::cout << "[WvmClassifier] : 'area' not right dim:" << nHK << "(==" << wvm->numLinFilters << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		if ((wvm->numFiltersPerLevel*wvm->numLevels)!=nHK){
			std::cout << "[WvmClassifier] : 'area' not right dim:" << nHK <<  "(==" << wvm->numLinFilters << ")" << std::endl;
			exit(EXIT_FAILURE);
		}

		wvm->area = new Area*[nHK];

		int cntval;
		for (hrsv=0;hrsv<nHK;hrsv++)  {

			mxArray* mval=mxGetField(pmxarray,hrsv,valstr);
 			if (mval == NULL ) {
				std::cout << "[WvmClassifier] : '" << valstr << "' not found (WVM: 'val_u', else: 'val', right *.mat/kernel?)" << std::endl;
				exit(EXIT_FAILURE);
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
		std::cout << "[WvmClassifier] : 'area' not found (right *.mat/kernel?)" << std::endl;
		exit(EXIT_FAILURE);
	}

	//read convolution of the appr. rsv (pp) in	mat file
	pmxarray = matGetVariable(pmatfile, "app_rsv_convol");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		const mwSize *dim = mxGetDimensions(pmxarray);
		int nHK=(int)dim[1];
		if (wvm->numLinFilters!=nHK){
			std::cout << "[WvmClassifier] : 'app_rsv_convol' not right dim:" << nHK << " (==" << wvm->numLinFilters << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		wvm->app_rsv_convol = new double[nHK];
		for (int hrsv=0; hrsv<nHK; ++hrsv)
			wvm->app_rsv_convol[hrsv]=matdata[hrsv]*65025.0; // because the training images grey level values were divided by 255;
		mxDestroyArray(pmxarray);
	} else {
		std::cout << "[WvmClassifier] : 'app_rsv_convol' not found" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (matClose(pmatfile) != 0) {
		std::cout << "[WvmClassifier] Error closing file" << std::endl;
	}
	std::cout << "[WvmClassifier] Done reading WVM!" << std::endl;


	
	//printf("fd_ReadDetector(): making the hierarchical thresholds\n");
	// making the hierarchical thresholds
	//MATFile *mxtFile = matOpen(args->threshold, "r");
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		std::cout << "[WvmClassifier] : Unable to open the file (wrong format?):" << std::endl << thresholdsFilename.c_str() << std::endl;
		exit(EXIT_FAILURE);
	} else {
		pmxarray = matGetVariable(pmatfile, "hierar_thresh");
		if (pmxarray == 0) {
			std::cout << "[WvmClassifier] : Unable to find the matrix hierar_thresh in the file " << thresholdsFilename.c_str() << std::endl;
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
	if(wvm->hierarchicalThresholdsFromFile.size() != wvm->numLinFilters) {
		std::cout << "[WvmClassifier] Something seems to be wrong, hierarchicalThresholdsFromFile.size() != numLinFilters; " << wvm->hierarchicalThresholdsFromFile.size() << "!=" << wvm->numLinFilters << std::endl;
		exit(EXIT_FAILURE);
	}
	wvm->setLimitReliabilityFilter(wvm->limitReliabilityFilter);	// This initializes the vector hierarchicalThresholds

	//for (i = 0; i < this->numLinFilters; ++i) printf("b%d=%g ",i+1,this->hierarchicalThresholds[i]);
	//printf("\n");
	std::cout << "[WvmClassifier] Done reading WVM-threshold file " << thresholdsFilename << std::endl;

	wvm->filter_output = new float[wvm->numLinFilters];
	wvm->u_kernel_eval = new float[wvm->numLinFilters];

	wvm->setNumUsedFilters(wvm->numUsedFilters);	// Makes sure that we don't use more filters than the loaded WVM has, and if zero, set to numLinFilters.

}


void WvmClassifier::setNumUsedFilters(int var)
{
	if(var>this->numLinFilters || var==0) {
		this->numUsedFilters = this->numLinFilters;
	} else {
		this->numUsedFilters = var;
	}
}

int WvmClassifier::getNumUsedFilters(void)
{
	return this->numUsedFilters;
}

void WvmClassifier::setLimitReliabilityFilter(float var)
{
	this->limitReliabilityFilter = var;
	if(var != 0.0f) {
		this->hierarchicalThresholds.clear();	// Maybe we could not use clear here and just assign the new value. But then we'd have a problem the first time we call this function.
		// But we could pre-allocate the right size.
		for (unsigned int i=0; i<this->hierarchicalThresholdsFromFile.size(); ++i) {
			this->hierarchicalThresholds.push_back(this->hierarchicalThresholdsFromFile[i] + limitReliabilityFilter);
		}
	} else {
		this->hierarchicalThresholds = this->hierarchicalThresholdsFromFile;
	}

}

float WvmClassifier::getLimitReliabilityFilter(void)
{
	return this->limitReliabilityFilter;
}

/*
 * WVM evaluation with a histogram equalized patch (64 bins) and patch integral image 
*/
float WvmClassifier::linEvalWvmHisteq64(
												int level, int n,  //n: n-th WSV at this apprlevel
												float* hk_kernel_eval,
												float* u_kernel_eval,
												const IImg* iimg_x/*=NULL*/, 
												const IImg* iimg_xx/*=NULL*/      ) const 
{
	/* iimg_x and iimg_xx are now patch-integral images! */

	float *this_weight = hkWeights[level];
	float res = -lin_thresholds[level];
	double norm = 0.0F;
	int p;
	const int fx = 0;
	const int fy = 0;
	const int lx = filter_size_x-1;
	const int ly = filter_size_y-1;


		
	//###########################################
	// compute evaluation and return it to fout.Pixel(x,y)
	// eval = sum_{i=1,...,level} [ hkWeights[level][i] * exp(-basisParam*(linFilters[i]-img.data)^2) ]
	//      = sum_{i=1,...,level} [ hkWeights[level][i] * exp(-basisParam*norm[i]) ]
	//      = sum_{i=1,...,level} [ hkWeights[level][i] * hk_kernel_eval[i] ]

	//===========================================
	// first, compute this kernel and save it in hk_kernel_eval[level], 
	// because the hk_kernel_eval[0,...,level-1] are the same for each new level
	// hk_kernel_eval[level] = exp( -basisParam * (linFilters[level]-img.data)^2 )
	//                       = exp( -basisParam * norm[level] )

	//rvm_kernel_begin = clock();
	//rvm_norm_begin = clock();


	/* adjust wvm calculation to patch-II */

	//.........................................
	// calculate the norm for that kernel (hk_kernel_eval[level] = exp(-basisParam*(linFilters[level]-img.data)^2))
	//  norm[level] = ||x-z||^2 = (linFilters[level]-img.data)^2   approximated by     
	//  norm[level] = ||x-p||^2 = x*x - 2*x*p + p*p   with  (x: cur. patch img.data, p: appr. RSV linFilters[level])

	double	norm_new = 0.0F,sum_xp = 0.0F,sum_xx = 0.0F,sum_pp = 0.0F;
	float	sumv = 0.0f,sumv0 = 0.0f;
	int		r,v,/*uur,uull,dll,dr,*/ax1,ax2,ay1,ay1w,ay2w;
	PRec	rec;


	//1st term: x'*x (integral image over x^2)
	//sxx_begin = clock();
	//norm_new=iimg_xx->ISumV(0,0,0,399,0,0,lx,ly);
	//norm_new=iimg_xx->ISum(fx,fy,lx,ly);
	//norm_new=iimg_xx->data[dr];
	//uur=(fy-1)*20/*img.w*/ + lx; uull=(fy-1)*20/*img.w*/ + fx-1; dll=ly*20/*img.w*/ + fx-1; dr=ly*20/*img.w*/ + lx;
	const int dr=ly*filter_size_x/*img.w*/ + lx;
	/*if (fx>0 && fy>0)  {
		norm_new= iimg_xx->data[dr] - iimg_xx->data[uur] - iimg_xx->data[dll] + iimg_xx->data[uull];
		sumv0=    iimg_x->data[dr]  - iimg_x->data[uur]  - iimg_x->data[dll]  + iimg_x->data[uull]; 
	} else if (fx>0)   {
		norm_new= iimg_xx->data[dr] - iimg_xx->data[dll]; sumv0= iimg_x->data[dr] - iimg_x->data[dll];
	} else if (fy>0)	{
	    norm_new= iimg_xx->data[dr] - iimg_xx->data[uur]; sumv0= iimg_x->data[dr] - iimg_x->data[uur];
	} else {*///if (fx==0 && fy==0)
	    norm_new= iimg_xx->data[dr]; sumv0= iimg_x->data[dr];
	//}
	sum_xx=norm_new;

	//Profiler.sxx += (double)(clock()-sxx_begin);

	//2nd term : 2x'*p 
	//    (sum 'sum_xp' over the sums 'sumv' for each gray level of the appr. RSV 
	//     over all rectangles of that gray level which are calculated by integral input image
	//     multiplied the sum_v by the gray value)
	//
	//		dh. sum_xp= sumv0*val0 - sum_{v=1}^cntval ( sum_{r=0}^cntrec_v(iimg_x->ISum(rec_{r,v})) * val_v )
	// also we can simplify 
	//		2x'*p = 2x'* ( sum_{l=0}^{lev-1}(res_{l,n}) + res_{lev,n} ) for the n-th SV at apprlevel lev
	//		      = 2x'*u_{lev-1,n} + 2x'*res_{lev,n} 
	//		      = u_kernel_eval[n] + sum_xp, with p=res_{lev,n}
	//                                        and u_kernel_eval[n]_{lev+1}=u_kernel_eval[n] + sum_xp

	//sxp_begin = clock();
	//sumv0=iimg_x->ISum(fx,fy,lx,ly);
	//sumv0=iimg_x->ISumV(0,0,0,399,0,0,lx,ly);
	//sumv0=iimg_x->data[dr];
	for (v=1;v<area[level]->cntval;v++) {
		sumv=0;
		for (r=0;r<area[level]->cntrec[v];r++)   {
			rec=&area[level]->rec[v][r];
			//sxp_iimg_begin = clock();
			//if (rec->x1==rec->x2 && rec->y1==rec->y2) 
			//	sumv+=img.data[(fy+rec->y1)*img.w+(fx+rec->x1)];
			//
			////TODO: fasten up for lines 
			////else if (rec->x1==rec->x2 || rec->y1==rec->y2)
			//
			//else //if (rec->x1!=rec->x2 && rec->y1!=rec->y2)
			//	sumv+=iimg_x->ISum(fx+rec->x1,fy+rec->y1,fx+rec->x2,fy+rec->y2);
			//	sumv+=iimg_x->ISumV(rec->uull,rec->uur,rec->dll,rec->dr,rec->x1,rec->y1,rec->x2,rec->y2);
			//	sumv+=   iimg_x->data[rec->dr]                   - ((rec->y1>0)? iimg_x->data[rec->uur]:0) 
			//		   - ((rec->x1>0)? iimg_x->data[rec->dll]:0) + ((rec->x1>0 && rec->y1>0)? iimg_x->data[rec->uull]:0);
			ax1=fx+rec->x1-1; ax2=fx+rec->x2; ay1=fy+rec->y1;
			ay1w=(ay1-1)*filter_size_x/*img.w*/; ay2w=(fy+rec->y2)*filter_size_x/*img.w*/; 
			if (ax1+1>0 && ay1>0)
				sumv+=   iimg_x->data[ay2w +ax2] - iimg_x->data[ay1w +ax2]
					   - iimg_x->data[ay2w +ax1] + iimg_x->data[ay1w +ax1];
			else if	(ax1+1>0)
				sumv+=   iimg_x->data[ay2w +ax2] - iimg_x->data[ay2w +ax1];
			else if	(ay1>0)
            	sumv+=   iimg_x->data[ay2w +ax2] - iimg_x->data[ay1w +ax2];
			else //if (ax1==0 && ay1==0)
            	sumv+=   iimg_x->data[ay2w +ax2];

			//Profiler.sxp_iimg += (double)(clock()-sxp_iimg_begin);
		}
		//sxp_mval_begin = clock();
		sumv0-=sumv;
		sum_xp+=sumv*area[level]->val[v];
		//Profiler.sxp_mval += (double)(clock()-sxp_mval_begin);
	}
	sum_xp+=sumv0*area[level]->val[0];
	sum_xp+=u_kernel_eval[n];
	u_kernel_eval[n]=sum_xp;  //update u_kernel_eval[n]

	norm_new-=2*sum_xp;
	//Profiler.sxp += (double)(clock()-sxp_begin);

	//3rd term: p'*p (convolution of the appr. RSV - constrant calculated at the training)
	//spp_begin = clock();
	sum_pp=app_rsv_convol[level]; // patrik: �berfl�ssig?
	norm_new+=app_rsv_convol[level];
	//Profiler.spp += (double)(clock()-spp_begin);

	norm=norm_new;

	//Profiler.rvm_norm += (double)(clock()-rvm_norm_begin);

	//.........................................
	// calculate  now this kernel and save it in hk_kernel_eval[level], 
	// hk_kernel_eval[level] = exp(-basisParam*(linFilters[level]-img.data)^2) ]

	hk_kernel_eval[level] = (float)(exp(-basisParam*norm)); //save it, because they 0...level-1 the same for each new level 

	//===========================================
	// second, sum over all the kernels to get the output
	// eval = sum_{i=1,...,level} [ hkWeights[level][i] * exp(-basisParam*(linFilters[i]-img.data)^2) ]
	//      = sum_{i=1,...,level} [ hkWeights[level][i] * hk_kernel_eval[i] ]

	for (p = 0; p <= level; ++p) //sum k=0...level = b_level,k * Kernel_k
		res += this_weight[p] * hk_kernel_eval[p];

	//Profiler.rvm_kernel += (double)(clock()-rvm_kernel_begin);

	return res;
}


WvmClassifier::Area::Area(void)
{
	cntval=0;
	val=NULL;
	rec=NULL;
	cntrec=NULL;
	cntallrec=0;
}

WvmClassifier::Area::Area(int cv, int *cr) : cntval(cv)
{
	int r,v;

	val = new double[cntval];
	cntrec = new int[cntval];
	cntallrec=0;
	for (v=0;v<cntval;v++) {
		cntrec[v]=cr[v];
		cntallrec+=cr[v];
	}
	rec = new TRec*[cntval];
	for (v=0;v<cntval;v++) {
		rec[v] = new TRec[cntrec[v]];
		for (r=0;r<cntrec[v];r++) { rec[v][r].x1=rec[v][r].y1=rec[v][r].x2=rec[v][r].y2=0; }
	}
}

WvmClassifier::Area::~Area(void)
{
	if (rec!=NULL) for (int v=0;v<cntval;v++) delete [] rec[v];
	if (rec!=NULL) delete [] rec;
	if (cntrec!=NULL) delete [] cntrec;
	if (val!=NULL) delete [] val;
}

void WvmClassifier::Area::dump(char *name="") {
	int r,v;  

	std::cout << std::endl << "area" << name << ": cntval:" << cntval << ", cntallrec:" << cntallrec << ", val:";
	for (v=0;v<cntval;v++) std::cout << " " << val[v];	//printf(" %1.4f",val[v]);
	std::cout << std::endl;

	for (v=0;v<cntval;v++) {
		for (r=0;r<cntrec[v];r++) { 
			//printf("r[%d][%d]:(%d,%d,%d,%d) ",v,r,rec[v][r].x1,rec[v][r].y1,rec[v][r].x2,rec[v][r].y2);
			std::cout << "r[" << v << "][" << r << "]:(" << rec[v][r].x1 << "," << rec[v][r].y1 << "," << rec[v][r].x2 << "," << rec[v][r].y2 << ") ";
			if ((r%5)==4) 
				std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}


} /* namespace classification */

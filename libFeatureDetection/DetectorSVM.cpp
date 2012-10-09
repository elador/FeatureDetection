#include "stdafx.h"
#include "DetectorSVM.h"

#include "MatlabReader.h"
#include "FdPatch.h"
#include "SLogger.h"

#include <iostream>
#include <cmath>

DetectorSVM::DetectorSVM(void)
{
	identifier = "DetectorSVM";
	numSV = 0;
	support = 0;
	alpha = 0;
	posterior_svm[0] = 0.0f;
	posterior_svm[1] = 0.0f;
	limit_reliability = 0.0f;
}


DetectorSVM::~DetectorSVM(void)
{
	if(support != 0) {
		for(int i = 0; i < numSV; ++i)
			delete[] support[i];
	}
	delete[] support;
	delete[] alpha;
}

bool DetectorSVM::classify(FdPatch* fp)
{
	//std::cout << "[DetSVM] Classifying!\n";

	// Check if the patch has already been classified by this detector! If yes, don't do it again.
	FoutMap::iterator it = fp->fout.find(this->getIdentifier());
	if(it!=fp->fout.end()) {	// The fout value is already in the map
								// Assumption: When fp->fout is not found, we assume that also fp->certainty is not yet set! This assumption should always hold.
		if(Logger->getVerboseLevelText()>=4) {
			std::cout << "[DetectorSVM] An element 'fout' already exists for this detector. 'fout' not changed. Not running the same detector twice over a patch." << std::endl;
		}
		if (it->second >= this->limit_reliability) {
			return true;
		} else {
			return false;
		}
	} // else: The fout value is not already computed. Go ahead.
	
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
}


float DetectorSVM::kernel(unsigned char* data, unsigned char* support, int nonLinType, float basisParam, float divisor, int polyPower, int nDim)
{
	int dot = 0;
	int val2;
	float out;
	float val;
	int i;
	switch (nonLinType) {
	case 1: // polynomial
		for (i = 0; i != nDim; ++i)
			dot += data[i] * support[i];
		out = (dot+basisParam)/divisor;
		val = out;
		for (i = 1; i < polyPower; i++)
			out *= val;
		return out;
	case 2: // RBF
		for (i = 0; i != nDim; ++i) {
			val2 = data[i] - support[i];
			dot += val2 * val2;
		}
		return (float)exp(-basisParam*dot);
	default: assert(0);
	}
	return 0;

}

int DetectorSVM::load(const std::string filename)
{
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] Loading " << filename << std::endl;
	}

	MatlabReader *configReader = new MatlabReader(filename);
	int id;
	char buff[255], key[255], pos[255];

	if(!configReader->getKey("FD.ffp", buff)) {	// which feature point does this detector detect?
		std::cout << "[DetSVM] Warning: Key in Config nicht gefunden, key:'" << "FD.ffp" << "'" << std::endl;
	} else {
		if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
			std::cout << "[DetSVM] ffp: " << atoi(buff) << std::endl;
		}
	}

//	if (!configReader->getKey("ALLGINFO.outputdir", this->outputPath)) // Output folder of this detector
//		std::cout << "[DetSVM] Warning: Key in Config nicht gefunden, key:'" << "ALLGINFO.outputdir" << "'" << std::endl;
//	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
//		std::cout << "[DetSVM] outputdir: " << this->outputPath << std::endl;
//	}

	//min. und max. erwartete Anzahl Gesichter im Bild (vorerst null bis eins);											  
	sprintf(pos,"FD.expected_number_faces.#%d",0);																		  
	if (!configReader->getKey(pos,buff))																						  
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:" << pos << ", nehme Default: "<< this->expected_num_faces[0] << std::endl;
	else
		this->expected_num_faces[0]=atoi(buff);
	sprintf(pos,"FD.expected_number_faces.#%d",1);
	if (!configReader->getKey(pos,buff))
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:" << pos << ", nehme Default: " << this->expected_num_faces[1] << std::endl;
	else
		this->expected_num_faces[1]=atoi(buff);

	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] expected_num_faces: " << this->expected_num_faces[0] << ", " << this->expected_num_faces[1] << std::endl;
	}

	//Grenze der Zuverlaesigkeit ab der Gesichter aufgenommen werden (Diffwert fr SVM-Schwelle)
	if (!configReader->getKey("FD.limit_reliability",buff))
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.limit_reliability', nehme Default: " << this->limit_reliability << std::endl;
	else this->limit_reliability=(float)atof(buff);

	//ROI: left, top, right, bottom
    // 0 0 0 0 (ganze Bild), -1 -1 -1 -1 (bzw. ganze FD-ROI) 
	int v=1;
	if (!configReader->getInt("FD.roi.#0",&v))		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.roi.#0', nehme Default: " << this->roi.left << std::endl;
	else										this->roi.left=v;
	if (!configReader->getInt("FD.roi.#1",&v))		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.roi.#1', nehme Default: " << this->roi.top << std::endl;
	else										this->roi.top=v;
	if (!configReader->getInt("FD.roi.#2",&v))		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.roi.#2', nehme Default: " << this->roi.right << std::endl;
	else										this->roi.right=v;
	if (!configReader->getInt("FD.roi.#3",&v))		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.roi.#3', nehme Default: " << this->roi.bottom << std::endl;
	else										this->roi.bottom=v;
	
	//Minimale Gesichtsoehe in Pixel 
	if (!configReader->getInt("FD.face_size_min",&this->subsamplingMinHeight))
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.face_size_min', nehme Default: " << this->subsamplingMinHeight << std::endl;
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] face_size_min: " << this->subsamplingMinHeight << std::endl;
	}
	//Anzahl der Skalierungen
	if (!configReader->getInt("FD.maxscales",&this->numSubsamplingLevels))
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.maxscales', nehme Default: " << this->numSubsamplingLevels << std::endl;
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] maxscales: " << this->numSubsamplingLevels << std::endl;
	}
	//Scalierungsfaktor 
	if (!configReader->getKey("FD.scalefactor",buff))
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.scalefactor', nehme Default: " << this->subsamplingFactor << std::endl;
	else
		this->subsamplingFactor=(float)atof(buff);
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] scalefactor: " << this->subsamplingFactor << std::endl;
	}

	//Kassifikator
	char fn_classifier[500];
	if (!configReader->getKey("FD.classificator", fn_classifier))
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.classificator', nehme Default: " << fn_classifier << std::endl;

	//Schwellwerte
	char fn_threshold[500];
	if (!configReader->getKey("FD.threshold", fn_threshold))
		std::cout << "[DetSVM] WARNING: Key in Config nicht gefunden, key:'FD.threshold', nehme Default: " << fn_threshold << std::endl;

	delete configReader;

	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[DetSVM] Loading " << fn_classifier << std::endl;
	}
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

	return 1;

}

int DetectorSVM::init_for_image(FdImage* img)
{
	initPyramids(img);
	initROI(img);
	return 1;
}

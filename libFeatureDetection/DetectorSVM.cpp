#include "stdafx.h"
#include "DetectorSVM.h"


DetectorSVM::DetectorSVM(void)// : faces(NULL)
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
	
	float res = -this->nonlin_threshold;	// TODO ASK MR Why minus???
	for (int i = 0; i != this->numSV; ++i) {
		res += this->alpha[i] * kernel(fp->data, this->support[i], this->nonLinType, this->basisParam, this->divisor, this->polyPower, this->filter_size_x*this->filter_size_y);
	}
	//fp->fout = res;
	std::pair<FoutMap::iterator, bool> fout_insert = fp->fout.insert(FoutMap::value_type(this->identifier, res));
	if(fout_insert.second == false) {
		std::cout << "[DetectorSVM] An element 'fout' already exists for this detector. 'fout' not changed. You ran the same detector twice over a patch." << std::endl;
	}
	//fp->certainty = 1.0f / (1.0f + exp(posterior_svm[0]*res + posterior_svm[1]));
	std::pair<CertaintyMap::iterator, bool> certainty_insert = fp->certainty.insert(CertaintyMap::value_type(this->identifier, 1.0f / (1.0f + exp(posterior_svm[0]*res + posterior_svm[1]))));
	if(fout_insert.second == false) {
		std::cout << "[DetectorSVM] An element 'fout' already exists for this detector. 'fout' not changed. You ran the same detector twice over a patch." << std::endl;
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

int DetectorSVM::load(const char* filename)
{
	//char* configFile = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat";
	std::cout << "[DetSVM] Loading " << filename << std::endl;

	MatlabReader *configReader = new MatlabReader(filename);
	int id;
	char buff[255], key[255], pos[255];

	if(!configReader->getKey("FD.ffp", buff))	// which feature point does this detector detect?
		std::cout << "Warning: Key in Config nicht gefunden, key:'" << "FD.ffp" << "'" << std::endl;
	else
		std::cout << "[DetSVM] ffp: " << atoi(buff) << std::endl;

	if (!configReader->getKey("ALLGINFO.outputdir", this->outputPath)) // Output folder of this detector
		std::cout << "Warning: Key in Config nicht gefunden, key:'" << "ALLGINFO.outputdir" << "'" << std::endl;
	std::cout << "[DetSVM] outputdir: " << this->outputPath << std::endl;

	//min. und max. erwartete Anzahl Gesichter im Bild (vorerst null bis eins);											  
	sprintf(pos,"FD.expected_number_faces.#%d",0);																		  
	if (!configReader->getKey(pos,buff))																						  
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n", pos,this->expected_num_faces[0]);
	else
		this->expected_num_faces[0]=atoi(buff);
	sprintf(pos,"FD.expected_number_faces.#%d",1);
	if (!configReader->getKey(pos,buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n", pos,this->expected_num_faces[1]);
	else
		this->expected_num_faces[1]=atoi(buff);

	std::cout << "[DetSVM] expected_num_faces: " << this->expected_num_faces[0] << ", " << this->expected_num_faces[1] << std::endl;

	//Grenze der Zuverlaesigkeit ab der Gesichter aufgenommen werden (Diffwert fr SVM-Schwelle)
	if (!configReader->getKey("FD.limit_reliability",buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %g\n",
		"FD.limit_reliability",this->limit_reliability);
	else this->limit_reliability=(float)atof(buff);

	//ROI: left, top, right, bottom
    // 0 0 0 0 (ganze Bild), -1 -1 -1 -1 (bzw. ganze FD-ROI) 
	int v=1;
	if (!configReader->getInt("FD.roi.#0",&v))		printf("WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n","FD.roi.#0",this->roi.left);
	else										this->roi.left=v;
	if (!configReader->getInt("FD.roi.#1",&v))		printf("WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n","FD.roi.#1",this->roi.top);
	else										this->roi.top=v;
	if (!configReader->getInt("FD.roi.#2",&v))		printf("WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n","FD.roi.#2",this->roi.right);
	else										this->roi.right=v;
	if (!configReader->getInt("FD.roi.#3",&v))		printf("WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n","FD.roi.#3",this->roi.bottom);
	else										this->roi.bottom=v;
	
	//Minimale Gesichtsoehe in Pixel 
	if (!configReader->getInt("FD.face_size_min",&this->subsamplingMinHeight))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n", "FD.face_size_min",this->subsamplingMinHeight);
	std::cout << "[DetSVM] face_size_min: " << this->subsamplingMinHeight << std::endl;
	//Anzahl der Skalierungen
	if (!configReader->getInt("FD.maxscales",&this->numSubsamplingLevels))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %d\n", "FD.maxscales",this->numSubsamplingLevels);
	std::cout << "[DetSVM] maxscales: " << this->numSubsamplingLevels << std::endl;
	//Scalierungsfaktor 
	if (!configReader->getKey("FD.scalefactor",buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %g\n", "FD.scalefactor",this->subsamplingFactor);
	else
		this->subsamplingFactor=(float)atof(buff);
	std::cout << "[DetSVM] scalefactor: " << this->subsamplingFactor << std::endl;

	//Kassifikator
	char fn_classifier[500];
	if (!configReader->getKey("FD.classificator", fn_classifier))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: '%s'\n",
		"FD.classificator", fn_classifier);

	//Schwellwerte
	char fn_threshold[500];
	if (!configReader->getKey("FD.threshold", fn_threshold))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: '%s'\n",
		"FD.threshold", fn_threshold);

	delete configReader;

	std::cout << "[DetSVM] Loading " << fn_classifier << std::endl;
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	double *matdata;
	pmatfile = matOpen(fn_classifier, "r");
	if (pmatfile == NULL) {
		std::cout << "Error: Opening file" << std::endl;
		return -1;
	}

	pmxarray = matGetVariable(pmatfile, "param_nonlin1");
	if (pmxarray == 0) {
		std::cout << "Error: There is a no param_nonlin1 in the file." << std::endl;
		return 0;
	}
	std::cout << "[DetSVM] Reading param_nonlin1" << std::endl;
	matdata = mxGetPr(pmxarray);
	this->nonlin_threshold = (float)matdata[0];
	this->nonLinType       = (int)matdata[1];
	this->basisParam       = (float)(matdata[2]/65025.0); // because the training image's graylevel values were divided by 255
	this->polyPower        = (int)matdata[3];
	this->divisor          = (float)matdata[4];
	mxDestroyArray(pmxarray);
		
	pmxarray = matGetVariable(pmatfile, "support_nonlin1");
	if (pmxarray == 0) {
		std::cout << "Error: There is a nonlinear SVM in the file, but the matrix support_nonlin1 is lacking!" << std::endl;
		return 0;
	} 
	if (mxGetNumberOfDimensions(pmxarray) != 3) {
		std::cout << "Error: The matrix support_nonlin1 in the file should have 3 dimensions." << std::endl;
		return 0;
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
		std::cout << "Error: There is a nonlinear SVM in the file but the matrix threshold_nonlin is lacking." << std::endl;
		return 0;
	}
	matdata = mxGetPr(pmxarray);
	for (is = 0; is < this->numSV; ++is)
		this->alpha[is] = (float)matdata[is];
	mxDestroyArray(pmxarray);

	if (matClose(pmatfile) != 0) {
		std::cout << "Error closing file" << std::endl;
	}

	
	pmatfile = matOpen(fn_threshold, "r");
	if (pmatfile == 0) {
		printf("fd_ReadDetector(): Unable to open the file (wrong format?):\n'%s' \n", fn_threshold);
		return 1;
	} else {
		//printf("fd_ReadDetector(): read posterior_svm parameter for probabilistic SVM output\n");
		//read posterior_wrvm parameter for probabilistic WRVM output
		//TODO is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
		pmxarray = matGetVariable(pmatfile, "posterior_svm");
		if (pmxarray == 0) {
			fprintf(stderr, "WARNING: Unable to find the vector posterior_svm, disable prob. SVM output;\n");
			this->posterior_svm[0]=this->posterior_svm[1]=0.0f;
		} else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			if (dim[1] != 2) {
				fprintf(stderr, "WARNING: Size of vector posterior_svm !=2, disable prob. SVM output;\n");
				this->posterior_svm[0]=this->posterior_svm[1]=0.0f;
			} else {
				this->posterior_svm[0]=(float)matdata[0]; this->posterior_svm[1]=(float)matdata[1];
			}
			mxDestroyArray(pmxarray);
		}
		matClose(pmatfile);
	}
	std::cout << "[DetSVM] Done reading posterior_svm [" << posterior_svm[0] << ", " << posterior_svm[1] << "] from threshold file " << fn_threshold << std::endl;

	std::cout << "[DetSVM] Done reading SVM!" << std::endl;

	this->stretch_fac = 255.0f/(float)(filter_size_x*filter_size_y);	// HistEq64 initialization

	return 1;

}

int DetectorSVM::init_for_image(FdImage* img)
{
	initPyramids(img);
	initROI(img);
	return 1;
}
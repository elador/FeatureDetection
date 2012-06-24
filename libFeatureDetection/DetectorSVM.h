#pragma once
#include "vdetectorvectormachine.h"
#include "FdPatch.h"
#include "MatlabReader.h"

class DetectorSVM : public VDetectorVectorMachine
{
public:
	DetectorSVM(void);
	~DetectorSVM(void);

	int load(const char*);
	int init_for_image(FdImage*);


protected:

	bool classify(FdPatch*);
	float kernel(unsigned char*, unsigned char*, int, float, float, int, int);

	int numSV;
	unsigned char** support;	// support[i] hold the support vector i
	float* alpha;				// alpha[i] hold the weight of the support vector i

	float posterior_svm[2];	// probabilistic svm output: p(ffp|t) = 1 / (1 + exp(p[0]*t +p[1]))

	float limit_reliability;	// if fout>=limit_reliability(threshold_fullsvm), then its a face. (MR default: -1.2)
};


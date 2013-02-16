#pragma once
#include "VDetectorVectorMachine.h"

class RegressorSVR : public VDetectorVectorMachine
{
public:
	RegressorSVR(void);
	~RegressorSVR(void);

	int load(const std::string);
	int initForImage(FdImage*);
	bool classify(FdPatch*);

protected:

	float kernel(unsigned char*, unsigned char*, int, float, float, int, int);

	int numSV;
	unsigned char** support;	// support[i] hold the support vector i
	float* alpha;				// alpha[i] hold the weight of the support vector i

};


#pragma once
#include "VDetector.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"

class CascadeWvmSvm : public VDetector
{
public:
	CascadeWvmSvm(void);
	~CascadeWvmSvm(void);

	CascadeWvmSvm(const std::string matfile);

	int initForImage(FdImage*);
	int detectOnImage(FdImage*);

	DetectorSVM *svm;
	DetectorWVM *wvm;

	std::vector<FdPatch*> candidates;
};


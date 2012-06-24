#pragma once
#include "VDetector.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"

class CascadeWvmSvm : public VDetector
{
public:
	CascadeWvmSvm(void);
	~CascadeWvmSvm(void);

	CascadeWvmSvm(char* matfile);

	int init_for_image(FdImage*);
	int detect_on_image(FdImage*);

	DetectorSVM *svm;
	DetectorWVM *wvm;

	std::vector<FdPatch*> candidates;
};


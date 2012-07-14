#pragma once
#include "VDetector.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "OverlapElimination.h"

class CascadeWvmOeSvmOe : public VDetector
{
public:
	CascadeWvmOeSvmOe(void);
	~CascadeWvmOeSvmOe(void);

	CascadeWvmOeSvmOe(const std::string matfile);

	int init_for_image(FdImage*);
	int detect_on_image(FdImage*);

	DetectorSVM *svm;
	DetectorWVM *wvm;
	OverlapElimination *oe;

	std::vector<FdPatch*> candidates;
};


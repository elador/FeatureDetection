#pragma once

#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "OverlapElimination.h"
#include "SLogger.h"

class CascadeWvmOeSvmOe
{
public:
	CascadeWvmOeSvmOe(void);
	~CascadeWvmOeSvmOe(void);

	CascadeWvmOeSvmOe(char* matfile);

	int init_for_image(FdImage*);
	int detect_on_image(FdImage*);

	DetectorSVM *svm;
	DetectorWVM *wvm;
	OverlapElimination *oe;

	std::vector<FdPatch*> candidates;
};


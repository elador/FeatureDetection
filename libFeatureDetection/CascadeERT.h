#pragma once
#include "VDetector.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "RegressorWVR.h"
#include "RegressorSVR.h"
#include "OverlapElimination.h"
#include "SLogger.h"

class CascadeERT : public VDetector
{
public:
	CascadeERT(void);
	~CascadeERT(void);

	int init_for_image(FdImage*);
	int detect_on_image(FdImage*);

	DetectorSVM *svm;
	DetectorWVM *wvm;
	OverlapElimination *oe;

	DetectorSVM *svml;
	DetectorWVM *wvml;
	OverlapElimination *oel;

	DetectorSVM *svmr;
	DetectorWVM *wvmr;
	OverlapElimination *oer;

	RegressorSVR *svr;
	RegressorWVR *wvr;


	std::vector<FdPatch*> candidates;
	
	//void scaleDown(std::vector<FdPatch*> patches);
	//void scaleUp(std::vector<FdPatch*> patches_to_scale_up, std::vector<FdPatch*> patchlist_original_size);

};


#pragma once
#include "VDetector.h"
#include "CascadeWvmOeSvmOe.h"
#include "CircleDetector.h"
#include "SkinDetector.h"
#include "CascadeFacialFeaturePointsSimple.h"

class CascadeFacialFeaturePoints : public VDetector
{
public:
	CascadeFacialFeaturePoints(void);
	~CascadeFacialFeaturePoints(void);

	//CascadeWvmOeSvmOe *face_frontal;
	//CascadeWvmOeSvmOe *eye_left;

	DetectorWVM *wvm_frontal;
	OverlapElimination *oe;

	CircleDetector *circleDet;
	SkinDetector *skinDet;

	CascadeFacialFeaturePointsSimple *ffpCasc;

	
	int init_for_image(FdImage*);
	int detect_on_image(FdImage*);

	std::vector<FdPatch*> candidates;
};


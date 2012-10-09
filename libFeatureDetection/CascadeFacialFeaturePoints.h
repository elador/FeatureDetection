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

	
	int initForImage(FdImage*);
	int detectOnImage(FdImage*);

	std::vector<FdPatch*> candidates;
};


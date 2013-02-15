#pragma once
#include "VDetector.h"

#include "CascadeWvmOeSvmOe.h"
#include "FeaturePointsModelRANSAC.h"

class CircleDetector;
class SkinDetector;
class CascadeFacialFeaturePointsSimple;

class CascadeFacialFeaturePoints : public VDetector
{
public:
	CascadeFacialFeaturePoints(void);
	~CascadeFacialFeaturePoints(void);

	DetectorWVM *wvm_frontal;
	OverlapElimination *oe;

	CircleDetector *circleDet;
	SkinDetector *skinDet;

	FeaturePointsRANSAC *ffpRansac;

	CascadeFacialFeaturePointsSimple *ffpCasc;

	int initForImage(FdImage*);
	int detectOnImage(FdImage*);

	std::vector<FdPatch*> candidates;
};


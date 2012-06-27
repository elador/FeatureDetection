#pragma once
#include "VDetector.h"
#include "CascadeWvmOeSvmOe.h"

class CascadeFacialFeaturePointsSimple : public VDetector
{
public:
	CascadeFacialFeaturePointsSimple(void);
	~CascadeFacialFeaturePointsSimple(void);

	int initForImage(FdImage*);
	int detectOnImage(FdImage*);

	CascadeWvmOeSvmOe *leye;
	CascadeWvmOeSvmOe *reye;
	CascadeWvmOeSvmOe *nosetip;
	CascadeWvmOeSvmOe *lmouth;
	CascadeWvmOeSvmOe *rmouth;

	std::vector<FdPatch*> candidates;
};


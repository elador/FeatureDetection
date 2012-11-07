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

	int initForImage(FdImage*);
	int detectOnImage(FdImage*);
	void setRoiInImage(Rect);	// Set the in-image ROI for all cascade stages, and overwrite the values set by reading the config file.

	DetectorSVM *svm;
	DetectorWVM *wvm;
	OverlapElimination *oe;

	std::vector<FdPatch*> candidates;

	void setIdentifier(std::string identifier);

};

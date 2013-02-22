#pragma once
#include "VDetector.h"
#include "CascadeWvmOeSvmOe.h"

#include <string>
#include <vector>

class CascadeFacialFeaturePointsSimple : public VDetector
{
public:
	CascadeFacialFeaturePointsSimple(void);
	~CascadeFacialFeaturePointsSimple(void);

	int initForImage(FdImage*);
	std::vector<std::pair<std::string, std::vector<FdPatch*> > > detectOnImage(FdImage*);
	void setRoiInImage(Rect);	// Set the in-image ROI for all feature classifiers, and overwrite the values set by reading the config file.

	std::map<std::string, CascadeWvmOeSvmOe*> detectors;	// Maybe it would be better to make a std::map<std::string, VDetector*> ? (and make a typedef)

	std::vector<FdPatch*> candidates;
};


#pragma once
#include "VDetector.h"

#include "utility.h"

#include <iostream>
#include <vector>

class VDetectorVectorMachine : public VDetector
{
public:
	VDetectorVectorMachine(void);
	virtual ~VDetectorVectorMachine(void);

	virtual int load(const std::string filename) = 0;
	virtual bool classify(FdPatch*) = 0;

	int initPyramids(FdImage*);	// img -> pyrs (save in img)
	std::vector<FdPatch*> detect_on_image(FdImage*);
	std::vector<FdPatch*> detect_on_patchvec(std::vector<FdPatch*>&);
	int extractToPyramids(FdImage*);	// all pyrs -> patches (save in img)
	std::vector<FdPatch*> getPatchesROI(FdImage*, int, int, int, int, int, int, std::string);

protected:
	int extractAndHistEq64(const Pyramid*, FdPatch*);	// private (one patch out of pyr)

	float nonlin_threshold;		// b parameter of the SVM
	int nonLinType;				// 2 = rbf (?)
	float basisParam;
	int polyPower;
	float divisor;

	int filter_size_x, filter_size_y;	// width and height of the detector patch

	int subsamplingMinHeight;
	int numSubsamplingLevels;
	float subsamplingFactor;

	int subsamplingLevelStart;
	int subsamplingLevelEnd;
	//float *subsampfac;
	std::map<int, float> subsampfac;

	int *pyramid_widths;

	unsigned char* LUT_bin; // lookup table for the histogram equalization
	float stretch_fac; // stretch factor for histogram equalizaiton

};

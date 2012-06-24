#pragma once
#include "vdetector.h"
#include "FdImage.h"
#include "Rect.h"
#include "utility.h"

class VDetectorVectorMachine : public VDetector
{
public:
	VDetectorVectorMachine(void);
	virtual ~VDetectorVectorMachine(void);

	virtual int load(const char* filename) = 0;
	virtual bool classify(FdPatch*) = 0;

	int initPyramids(FdImage*);	// img -> pyrs (save in img)
	std::vector<FdPatch*> detect_on_image(FdImage*);
	std::vector<FdPatch*> detect_on_patchvec(std::vector<FdPatch*>&);
	int extract(FdImage*);	// all pyrs -> patches (save in img)
	std::vector<FdPatch*> extractPatches(FdImage*, std::vector<FdPatch*>&);

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
	float *subsampfac;

	int *pyramid_widths;

	unsigned char* LUT_bin; // lookup table for the histogram equalization
	float stretch_fac; // stretch factor for histogram equalizaiton

};

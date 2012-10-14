#pragma once

#include <string>
#include "Rect.h"

//class Rect;
class FdImage;

class VDetector
{
public:
	VDetector(void);
	virtual ~VDetector(void);

	std::string getIdentifier();
	void setIdentifier(std::string);

	//char outputPath[255];	// TODO: Delete this / use std::string
	int expected_num_faces[2];

	Rect roiDistFromBorder;
	Rect roiInImg;

private:
	bool canOutputProbabilistic;

protected:
	std::string identifier;
	int initROI(FdImage*);	// read the ROI from the Matlab config (roiDistFromBorder) and convert to roiInImg
};

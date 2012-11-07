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

	std::string getIdentifier(void);
	void setIdentifier(std::string);

	//char outputPath[255];	// TODO: Delete this / use std::string
	int expected_num_faces[2];

	Rect roiDistFromBorder;
	Rect roiInImg;

	int initROI(FdImage*);	// Convert roiDistFromBorder to roiInImg, given the FdImage. The roiDistFromBorder values come from the config file.

	void setRoiInImage(Rect);	// Set the in-image ROI, and overwrite the values set by reading the config file.

private:
	bool canOutputProbabilistic;

protected:
	std::string identifier;
};

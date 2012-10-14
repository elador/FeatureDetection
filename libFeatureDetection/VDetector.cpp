#include "stdafx.h"
#include "VDetector.h"

#include "SLogger.h"

#include "Rect.h"
#include "FdImage.h"

VDetector::VDetector(void)
{
	canOutputProbabilistic = false;
	expected_num_faces[0]=0; expected_num_faces[1]=1;
	//strcpy(outputPath, "");
	identifier = "Detector";
}


VDetector::~VDetector(void)
{
}


int VDetector::initROI(FdImage* img)
{
	//Set ROI for fd
	// 0 0 0 0 (fullimg), -1 -1 -1 -1 (full_fd_roi), else rel. to fd_box_ul 
	if ( (this->roiDistFromBorder==Rect(0, 0, 0, 0)) || (this->roiDistFromBorder==Rect(-1, -1, -1, -1))) {
		this->roiInImg=Rect(0, 0, 0, 0);
		this->roiInImg.bottom=img->h;
		this->roiInImg.right=img->w; 
	} else {
		this->roiInImg.left=this->roiDistFromBorder.left; 
		this->roiInImg.right=img->w - this->roiDistFromBorder.right; 
		this->roiInImg.top=this->roiDistFromBorder.top; 
		this->roiInImg.bottom=img->h - this->roiDistFromBorder.bottom;
	}
	return 1;

}

std::string VDetector::getIdentifier()
{
	return this->identifier;
}

void VDetector::setIdentifier(std::string identifier)
{
	this->identifier = identifier;
}

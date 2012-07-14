#include "stdafx.h"
#include "VDetector.h"

#include "SLogger.h"

#include "Rect.h"
#include "FdImage.h"

VDetector::VDetector(void)
{
	canOutputProbabilistic = false;
	expected_num_faces[0]=0; expected_num_faces[1]=1;
	strcpy(outputPath, "");
	identifier = "Detector";
}


VDetector::~VDetector(void)
{
}


int VDetector::initROI(FdImage* img)
{
	//Set ROI for fd
	// 0 0 0 0 (fullimg), -1 -1 -1 -1 (full_fd_roi), else rel. to fd_box_ul 
	if ( (this->roi==Rect(0, 0, 0, 0)) || (this->roi==Rect(-1, -1, -1, -1))) {
		this->roi_inImg=Rect(0, 0, 0, 0);
		this->roi_inImg.bottom=img->h;
		this->roi_inImg.right=img->w; 
	} else {
		this->roi_inImg.left=this->roi.left; 
		this->roi_inImg.right=img->w - this->roi.right; 
		this->roi_inImg.top=this->roi.top; 
		this->roi_inImg.bottom=img->h - this->roi.bottom;
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

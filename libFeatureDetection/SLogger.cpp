#include "StdAfx.h"
#include "SLogger.h"


SLogger::SLogger(void)
{
}


SLogger::~SLogger(void)
{
}

SLogger* SLogger::Instance(void)
{
	static SLogger instance;
	return &instance;
}

void SLogger::drawBoxes(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor, bool allocateNew, std::string filename)
{
	cv::Mat outimg;
	if(allocateNew == true) {
		outimg = rgbimg.clone();	// leave input image intact
	} else {
		outimg = rgbimg;	// doesnt copy
	}
	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		//if((*pit)->certainty["DetectorSVM"] > 0.7) {
			cv::rectangle(outimg, cv::Point((*pit)->c.x-(*pit)->w_inFullImg/2, (*pit)->c.y-(*pit)->h_inFullImg/2), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), cv::Scalar(0, 0, (float)255*(*pit)->certainty[identifierForColor]));
		//}
	}

	if(allocateNew == true && filename.empty()) {
		//doesnt make sense to allocate new img and then not save it. It will be gone.
	} else if(allocateNew == true && !filename.empty()) {
		cv::imwrite(filename, outimg);
	} else if(allocateNew == false && filename.empty()) {
		//i just write the boxes into the image, can be used/saved later
	} else {
		cv::imwrite(filename, outimg);
	}

}

void SLogger::drawYawAngle(cv::Mat rgbimg, std::vector<FdPatch*> patches)
{

	std::ostringstream text;
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 0.7;
	int thickness = 1;  
	int count=0;
	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		if((*pit)->certainty["DetectorSVM"] > 0.98) {
			text << "yaw=" << std::fixed << std::setprecision(1) << (*pit)->fout["RegressorSVR"]  << std::ends;
			cv::putText(rgbimg, text.str(), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), fontFace, fontScale, cv::Scalar::all(0), thickness,8);
			text.str("");
			count++;
			if(count>=2)
				break;
		}
	}

}

void SLogger::drawBoxesWithYawAngleColor(cv::Mat rgbimg, std::vector<FdPatch*> patches)
{

	cv::Scalar color;
	double yaw;
	float r, g, b;
	float m = (255.0f/45.0f);

	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		//if(((*pit)->fout["RegressorWVR"] > 25.0 || (*pit)->fout["RegressorWVR"] < -25.0) && (*pit)->c.s==0) {
			 yaw = (*pit)->fout["RegressorWVR"] + 90.0;
			 //float r, g, b;
			 //float m = (255.0f/45.0f);
			 if(yaw >= 0 && yaw < 45) {
				 r = 255;
				 g = m*yaw;
				 b = 0;
			 } else if(yaw >= 45 && yaw < 90) {
				 r = -m*(yaw-45.0)+255.0;
				 g = 255;
				 b = 0;
			 } else if(yaw >= 90 && yaw < 135) {
				 r = 0;
				 g = 255;
				 b = m*(yaw-90.0);
			 }else if(yaw >= 135 && yaw <= 180) {
				 r = 0;
				 g = -m*(yaw-135.0)+255.0;
				 b = 255;
			 } else {
				 std::cout << "Yaw>90 detected" << std::endl;
				 r=g=b=0;
			 }
			 color = cv::Scalar(b, g, r);
			 
			cv::rectangle(rgbimg, cv::Point((*pit)->c.x-(*pit)->w_inFullImg/2, (*pit)->c.y-(*pit)->h_inFullImg/2), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), color);
		//}
	}

}

void SLogger::drawCenterpointsWithYawAngleColor(cv::Mat rgbimg, std::vector<FdPatch*> patches, int scale)
{

	cv::Scalar color;
	double yaw;
	float r, g, b;
	float m = (255.0f/45.0f);

	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		if((*pit)->c.s==scale) {
			 yaw = (*pit)->fout["RegressorWVR"] + 90.0;
			 //float r, g, b;
			 //float m = (255.0f/45.0f);
			 if(yaw >= 0 && yaw < 45) {
				 r = 255;
				 g = m*yaw;
				 b = 0;
			 } else if(yaw >= 45 && yaw < 90) {
				 r = -m*(yaw-45.0)+255.0;
				 g = 255;
				 b = 0;
			 } else if(yaw >= 90 && yaw < 135) {
				 r = 0;
				 g = 255;
				 b = m*(yaw-90.0);
			 }else if(yaw >= 135 && yaw <= 180) {
				 r = 0;
				 g = -m*(yaw-135.0)+255.0;
				 b = 255;
			 } else {
				 std::cout << "Yaw>90 detected" << std::endl;
				 r=g=b=0;
			 }
			 color = cv::Scalar(b, g, r);
			 
			cv::rectangle(rgbimg, cv::Point((*pit)->c.x-1, (*pit)->c.y-1), cv::Point((*pit)->c.x+1, (*pit)->c.y+1), color);
		}
	}

}




void SLogger::drawPatchYawAngleColor(cv::Mat rgbimg, FdPatch patch)
{

	cv::Scalar color;
			 double yaw = patch.fout["RegressorWVR"] + 90.0;
			 float r, g, b;
			 float m = (255.0f/45.0f);
			 if(yaw >= 0 && yaw < 45) {
				 r = 255;
				 g = m*yaw;
				 b = 0;
			 } else if(yaw >= 45 && yaw < 90) {
				 r = -m*(yaw-45.0)+255.0;
				 g = 255;
				 b = 0;
			 } else if(yaw >= 90 && yaw < 135) {
				 r = 0;
				 g = 255;
				 b = m*(yaw-90.0);
			 }else if(yaw >= 135 && yaw <= 180) {
				 r = 0;
				 g = -m*(yaw-135.0)+255.0;
				 b = 255;
			 } else {
				 std::cout << "Yaw>90 detected" << std::endl;
				 r=g=b=0;
			 }
			 color = cv::Scalar(b, g, r);
			 
			cv::rectangle(rgbimg, cv::Point(0, 0), cv::Point(patch.w-1, patch.h-1), color);


}
#include "StdAfx.h"
#include "SLogger.h"


SLogger::SLogger(void)
{
	this->imgLogBasepath = "out\\";
}


SLogger::~SLogger(void)
{
}

SLogger* SLogger::Instance(void)
{
	static SLogger instance;
	return &instance;
}

void SLogger::setVerboseLevelText(int level)
{
	this->verboseLevelText = level;
}

void SLogger::setVerboseLevelImages(int level)
{
	this->verboseLevelImages = level;
}


void SLogger::LogImgInputGray(const cv::Mat *img, std::string filename)
{
	if(this->global.img.writeImgInputGray) {
		size_t idx;
		idx = filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << filename.substr(idx+1) << "_0InputGray.png" << std::ends;
		cv::imwrite(fn.str(), *img);
	}
}

void SLogger::LogImgInputRGB(const cv::Mat *img, std::string filename)
{
	if(this->global.img.writeImgInputRGB) {
		size_t idx;
		idx = filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << filename.substr(idx+1) << "_0InputRGB.png" << std::ends;
		cv::imwrite(fn.str(), *img);
	}
}

void SLogger::LogImgPyramid(const Pyramid* pyr, std::string filename, int pyrIdx)
{
	if(this->global.img.writeImgPyramids) {
		size_t idx;
		idx = filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << filename.substr(idx+1) << "_Pyramid" << pyrIdx << "_w" << pyr->w << ".png" << std::ends;
		pyr->writePNG(fn.str());
	}
}

void SLogger::LogImgDetectorCandidates(const FdImage* img, std::vector<FdPatch*> candidates, std::string detectorIdForCertaintyColor, std::string filenameAppend)
{
	if((this->global.img.writeDetectorCandidates==true) || this->verboseLevelImages>=2) {
		if(filenameAppend=="") {
			filenameAppend = detectorIdForCertaintyColor;
		} else {
			filenameAppend = detectorIdForCertaintyColor + "_" + filenameAppend;
		}
		size_t idx;
		idx = img->filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << img->filename.substr(idx+1) << "_" << filenameAppend << ".png" << std::ends;

		cv::Mat outimg;
		outimg = img->data_matbgr.clone();	// leave input image intact
		this->drawBoxesWithCertainty(outimg, candidates, detectorIdForCertaintyColor);

		cv::imwrite(fn.str(), outimg);
	}
}

void SLogger::LogImgDetectorFinal(const FdImage* img, std::vector<FdPatch*> candidates, std::string detectorIdForCertaintyColor, std::string filenameAppend)
{
	if((this->global.img.writeDetectorCandidates==true) || this->verboseLevelImages>=1) {
		if(filenameAppend=="") {
			filenameAppend = detectorIdForCertaintyColor;
		} else {
			filenameAppend = detectorIdForCertaintyColor + "_" + filenameAppend;
		}
		size_t idx;
		idx = img->filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << img->filename.substr(idx+1) << "_" << filenameAppend << ".png" << std::ends;

		cv::Mat outimg;
		outimg = img->data_matbgr.clone();	// leave input image intact
		this->drawBoxesWithCertainty(outimg, candidates, detectorIdForCertaintyColor);

		cv::imwrite(fn.str(), outimg);
	}
}

void SLogger::LogImgRegressor(const FdImage* img, std::vector<FdPatch*> candidates, std::string regressorIdForFoutAngleColor, std::string filenameAppend)
{
	if((this->global.img.writeRegressor==true) || this->verboseLevelImages>=1) {
		if(filenameAppend=="") {
			filenameAppend = regressorIdForFoutAngleColor;
		} else {
			filenameAppend = regressorIdForFoutAngleColor + "_" + filenameAppend;
		}
		size_t idx;
		idx = img->filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << img->filename.substr(idx+1) << "_" << filenameAppend << ".png" << std::ends;

		cv::Mat outimg;
		outimg = img->data_matbgr.clone();	// leave input image intact
		this->drawBoxesWithAngleColor(outimg, candidates, regressorIdForFoutAngleColor);

		if(this->global.img.drawRegressorFoutAsText==true) {
			this->drawFoutAsText(outimg, candidates, regressorIdForFoutAngleColor);
		}
		if(this->global.img.drawScales==true) {
			if(candidates.size()>0) {	// TODO this could be improved. I somehow need to reach the Detector.filtersize here... e.g. each Det attaches himself to the logger?
				this->drawAllScaleBoxes(outimg, &img->pyramids, regressorIdForFoutAngleColor, candidates[0]->w, candidates[0]->h);
			} else {
				std::cout << "[Logger] Warning: Can't draw the scales because the candidates-list is empty." << std::endl;
			}
		}

		cv::imwrite(fn.str(), outimg);
	}
}

void SLogger::LogImgRegressorPyramids(const FdImage* img, std::vector<FdPatch*> candidates, std::string regressorIdForFoutAngleColor, std::string filenameAppend)
{
	if((this->global.img.writeRegressorPyramids==true) || this->verboseLevelImages>=2) {
		if(filenameAppend=="") {
			filenameAppend = regressorIdForFoutAngleColor;
		} else {
			filenameAppend = regressorIdForFoutAngleColor + "_" + filenameAppend;
		}
		size_t idx;
		idx = img->filename.find_last_of("/\\");
		std::ostringstream fn;
		cv::Mat outimg;

		for(PyramidMap::const_iterator it = img->pyramids.begin(); it != img->pyramids.end(); ++it) {	// loop over all pyramids
			std::set<std::string>::const_iterator sit = it->second->detectorIds.find(regressorIdForFoutAngleColor);
			if(sit != it->second->detectorIds.end()) {	// if the current pyramid is used by the detector
				fn << this->imgLogBasepath << img->filename.substr(idx+1) << "_" << filenameAppend << "_w" << it->second->w << ".png" << std::ends;	// print it
				outimg = img->data_matbgr.clone();	// leave input image intact
				this->drawCenterpointsWithAngleColor(outimg, candidates, regressorIdForFoutAngleColor, it->second->w);

				if(this->global.img.drawScales==true) {
					if(candidates.size()>0) {	// TODO this could be improved. I somehow need to reach the Detector.filtersize here... e.g. each Det attaches himself to the logger?
						int pyrInOrigImg_w = (int)(((float)img->w/(float)it->second->w)*(float)candidates[0]->w+0.5f);
						int pyrInOrigImg_h = (int)(((float)img->h/(float)it->second->h)*(float)candidates[0]->h+0.5f);
						this->drawScaleBox(outimg, pyrInOrigImg_w, pyrInOrigImg_h);
					} else {
						std::cout << "[Logger] Warning: Can't draw the scales because the candidates-list is empty." << std::endl;
					}
				}

				cv::imwrite(fn.str(), outimg);
				fn.str("");
			}
		}
	}
}

void SLogger::drawBoxesWithCertainty(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor)
{
	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
			cv::rectangle(rgbimg, cv::Point((*pit)->c.x-(*pit)->w_inFullImg/2, (*pit)->c.y-(*pit)->h_inFullImg/2), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), cv::Scalar(0, 0, (float)255*(*pit)->certainty[identifierForColor]));
	}
}

void SLogger::drawBoxesWithAngleColor(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor)
{
	cv::Scalar color;
	double yaw;
	float r, g, b;
	float m = (255.0f/45.0f);

	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		yaw = (*pit)->fout[identifierForColor] + 90.0;
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
			//std::cout << "Yaw>90 detected" << std::endl;
			r=g=b=0;
		}
		color = cv::Scalar(b, g, r);			 
		cv::rectangle(rgbimg, cv::Point((*pit)->c.x-(*pit)->w_inFullImg/2, (*pit)->c.y-(*pit)->h_inFullImg/2), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), color);
	}
}

void SLogger::drawCenterpointsWithAngleColor(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor, int scale)
{
	cv::Scalar color;
	double yaw;
	float r, g, b;
	float m = (255.0f/45.0f);

	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		if((*pit)->c.s==scale) {
			 yaw = (*pit)->fout[identifierForColor] + 90.0;
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
				 //std::cout << "Yaw>90 detected" << std::endl;
				 r=g=b=0;
			 }
			color = cv::Scalar(b, g, r);
			cv::rectangle(rgbimg, cv::Point((*pit)->c.x-1, (*pit)->c.y-1), cv::Point((*pit)->c.x+1, (*pit)->c.y+1), color);
		}
	}
}

void SLogger::drawFoutAsText(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForFout)
{
	std::ostringstream text;
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 0.7;
	int thickness = 1;  
	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		text << "yaw=" << std::fixed << std::setprecision(1) << (*pit)->fout[identifierForFout]  << std::ends;
		cv::putText(rgbimg, text.str(), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
		text.str("");
	}

}

void SLogger::drawAllScaleBoxes(cv::Mat rgbimg, const PyramidMap* pyramids, std::string detectorId, int det_filtersize_x, int det_filtersize_y)
{
	for(PyramidMap::const_iterator it = pyramids->begin(); it != pyramids->end(); ++it) {	// loop over all pyramids
		std::set<std::string>::const_iterator sit = it->second->detectorIds.find(detectorId);
		if(sit != it->second->detectorIds.end()) {	// if the current pyramid is used by the detector
			int pyrInOrigImg_w = (int)(((float)rgbimg.cols/(float)it->second->w)*(float)det_filtersize_x+0.5f);
			int pyrInOrigImg_h = (int)(((float)rgbimg.rows/(float)it->second->h)*(float)det_filtersize_y+0.5f);
			this->drawScaleBox(rgbimg, pyrInOrigImg_w, pyrInOrigImg_h);
		}
	}
}

void SLogger::drawScaleBox(cv::Mat rgbimg, int w, int h)
{
	cv::Scalar color = cv::Scalar(175, 175, 175);	//cv::Scalar(b, g, r);
	cv::rectangle(rgbimg, cv::Point(0, 0), cv::Point(w-1, h-1), color);
}





/* Used once for patches, may be useful some time.
void SLogger::drawSinglePatchYawAngleColor(cv::Mat rgbimg, FdPatch patch)
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
*/

/* old, but maybe useful later
void SLogger::drawBoxesAndSaveOrDontSave(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor, bool allocateNew, std::string filename)
{
	cv::Mat outimg;
	if(allocateNew == true) {
		outimg = rgbimg.clone();	// leave input image intact
	} else {
		outimg = rgbimg;	// doesnt copy
	}
	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
			cv::rectangle(outimg, cv::Point((*pit)->c.x-(*pit)->w_inFullImg/2, (*pit)->c.y-(*pit)->h_inFullImg/2), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), cv::Scalar(0, 0, (float)255*(*pit)->certainty[identifierForColor]));
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
} */

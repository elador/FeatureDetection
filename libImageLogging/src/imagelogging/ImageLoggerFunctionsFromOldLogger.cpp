/*
 * ImageLogger.cpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */

#include "imagelogging/ImageLoggerFunctionsFromOldLogger.hpp"

namespace imagelogging {
/*
ImageLogger::ImageLogger()
{

	// TODO: Make the color change with the certainty. c=... See old lib fd_patch.h drawCircle(...)
	// int c=max(0,(min(255,(int)(255* (certainty-max_col[0])/(-max_col[0]+max_col[1])))));
	// c = 0..255
	// const float MAX_COL[2]={-1.0f,2.0f};			//Color range (value of which color is 0 and 255)
	// const float MAX_COL_PROB[2]={0.0f,1.0f};		//Color range for probabilities (value of which color is 0 and 255)

	landmarkData currLm;
	std::string currLmName = "reye_c";
	currLm.bgrColor = cv::Scalar(0.0f, 0.0f, 1.0f);		// red (0, 0, c)
	/*std::array<bool, 9> tmp		= {	false, true, false,
										false, true, true,
										false, false, false };*/	// TODO: Change this back to std::array later!!!
/*
	bool tmp[]					= {	false, true, false,
									false, true, true,
									false, false, false };
	const int TotalItems = sizeof(tmp)/sizeof(tmp[0]);
	std::vector<bool> vtmp(tmp, tmp+TotalItems);
	currLm.symbol = vtmp;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));
	
	currLmName = "leye_c";
	currLm.bgrColor = cv::Scalar(1.0f, 0.0f, 0.0f);		// blue (c, 0, 0)
	bool tmp2[]	= {	false, true, false,
									true, true, false,
									false, false, false };
	const int TotalItems2 = sizeof(tmp2)/sizeof(tmp2[0]);
	std::vector<bool> vtmp2(tmp2, tmp2+TotalItems2);
	currLm.symbol = vtmp2;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "nose_tip";
	currLm.bgrColor = cv::Scalar(0.0f, 1.0f, 0.0f);		// green (0, c, 0)
	bool tmp3[]	= {	false, false, false,
									false, true, false,
									true, false, true };
	const int TotalItems3 = sizeof(tmp3)/sizeof(tmp3[0]);
	std::vector<bool> vtmp3(tmp3, tmp3+TotalItems3);
	currLm.symbol = vtmp3;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "mouth_rc";
	currLm.bgrColor = cv::Scalar(0.0f, 1.0f, 1.0f);		// yellow (0, c, c)
	bool tmp4[]	= {	false, false, true,
									false, true, false,
									false, false, true };
	const int TotalItems4 = sizeof(tmp4)/sizeof(tmp4[0]);
	std::vector<bool> vtmp4(tmp4, tmp4+TotalItems4);
	currLm.symbol = vtmp4;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = -1.0f/6.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "mouth_lc";
	currLm.bgrColor = cv::Scalar(1.0f, 0.0f, 1.0f);		// magenta (violet) (c, 0, c)
	bool tmp5[]	= {	true, false, false,
									false, true, false,
									true, false, false };
	const int TotalItems5 = sizeof(tmp5)/sizeof(tmp5[0]);
	std::vector<bool> vtmp5(tmp5, tmp5+TotalItems5);
	currLm.symbol = vtmp5;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 1.0f/6.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "reye_oc";
	currLm.bgrColor = cv::Scalar(0.0f, 0.0f, 0.48f);		// orange (0, (int)((float)c/255.0f*122.0f), c)
	bool tmp6[]	= {	false, true, false,
									false, true, true,
									false, true, false };
	const int TotalItems6 = sizeof(tmp6)/sizeof(tmp6[0]);
	std::vector<bool> vtmp6(tmp6, tmp6+TotalItems6);
	currLm.symbol = vtmp6;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "leye_oc";
	currLm.bgrColor = cv::Scalar(1.0f, 1.0f, 0.0f);		// cyan (tuerkis) (c, c, 0)
	bool tmp7[]	= {	false, true, false,
									true, true, false,
									false, true, false };
	const int TotalItems7 = sizeof(tmp7)/sizeof(tmp7[0]);
	std::vector<bool> vtmp7(tmp7, tmp7+TotalItems7);
	currLm.symbol = vtmp7;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "mouth_ulb";
	currLm.bgrColor = cv::Scalar(0.63f, 0.75f, 0.9f);		// pink (MR PhD says: beige)
	bool tmp8[]	= {	false, false, false,	// //color_r=(int)((float)c/255.0f*145.0f);;  color_b=(int)((float)c/255.0f*200.0f);	color_g=(int)((float)c/255.0f*180.0f); break; //stahlgrau
															//color_r=(int)((float)c/255.0f*230.0f);;  color_b=(int)((float)c/255.0f*160.0f);	color_g=(int)((float)c/255.0f*190.0f); break; //pink
									true, true, true,
									false, true, false };
	const int TotalItems8 = sizeof(tmp8)/sizeof(tmp8[0]);
	std::vector<bool> vtmp8(tmp8, tmp8+TotalItems8);
	currLm.symbol = vtmp8;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "nose_rc";		// nosetrill_r TODO: nose_rc IS PROBABLY NOT THE RIGHT EAR LANDMARK IN THE 3DMM! CHECK! Might be nosetrill_r or any other!
	currLm.bgrColor = cv::Scalar(0.27f, 0.27f, 0.67f);		// brown
	bool tmp9[]	= {	true, false, false,	//color_r=(int)((float)c/255.0f*170.0f);  color_b=(int)((float)c/255.0f*70.0f);	color_g=(int)((float)c/255.0f*70.0f); break; //brown
									true, true, true,
									false, false, false };
	const int TotalItems9 = sizeof(tmp9)/sizeof(tmp9[0]);
	std::vector<bool> vtmp9(tmp9, tmp9+TotalItems9);
	currLm.symbol = vtmp9;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "nose_lc";		// nosetrill_l TODO: nose_lc IS PROBABLY NOT THE RIGHT EAR LANDMARK IN THE 3DMM! CHECK! Might be nosetrill_l or any other!
	currLm.bgrColor = cv::Scalar(0.04f, 0.78f, 0.69f);		// lemon
	bool tmp10[]	= {	false, false, true,	//color_r=(int)((float)c/255.0f*177.0f);  color_b=(int)((float)c/255.0f*10.0f);	color_g=(int)((float)c/255.0f*200.0f); break; //mint
									true, true, true,
									false, false, false };
	const int TotalItems10 = sizeof(tmp10)/sizeof(tmp10[0]);
	std::vector<bool> vtmp10(tmp10, tmp10+TotalItems10);
	currLm.symbol = vtmp10;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "rear_c";	// TODO THIS IS PROBABLY NOT THE RIGHT EAR LANDMARK IN THE 3DMM! CHECK!
	currLm.bgrColor = cv::Scalar(1.0f, 0.0f, 0.52f);		// blue-violet
	bool tmp11[]	= {	false, true, true,	//color_r=(int)((float)c/255.0f*132.0f);  color_b=(int)((float)c/255.0f*255.0f);	color_g=(int)((float)c/255.0f*0.0f); break; //blue-violet
									false, true, false,
									false, true, true };
	const int TotalItems11 = sizeof(tmp11)/sizeof(tmp11[0]);
	std::vector<bool> vtmp11(tmp11, tmp11+TotalItems11);
	currLm.symbol = vtmp11;
	currLm.displacementFactorH = 3.0f/10.0f; currLm.displacementFactorW = -1.0f/6.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));

	currLmName = "lear_c";	// TODO THIS IS PROBABLY NOT THE RIGHT EAR LANDMARK IN THE 3DMM! CHECK!
	currLm.bgrColor = cv::Scalar(0.0f, 0.6f, 0.0f);		// pale green
	bool tmp12[]	= {	true, true, false,	//color_r=(int)((float)c/255.0f*0.0f);  color_b=(int)((float)c/255.0f*153.0f);	color_g=(int)((float)c/255.0f*255.0f); break; //blasses gruen
									false, true, false,
									true, true, false };
	const int TotalItems12 = sizeof(tmp12)/sizeof(tmp12[0]);
	std::vector<bool> vtmp12(tmp12, tmp12+TotalItems12);
	currLm.symbol = vtmp12;
	currLm.displacementFactorH = 3.0f/10.0f; currLm.displacementFactorW = 1.0f/6.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));
	
	currLmName = "UNKNOWN";
	currLm.bgrColor = cv::Scalar(0.35f, 0.35f, 0.35f);		// gray
	bool tmp13[]	= {	true, false, true,
									false, true, false,
									true, false, true };
	const int TotalItems13 = sizeof(tmp13)/sizeof(tmp13[0]);
	std::vector<bool> vtmp13(tmp13, tmp13+TotalItems13);
	currLm.symbol = vtmp13;
	currLm.displacementFactorH = 0.0f; currLm.displacementFactorW = 0.0f;
	this->landmarksData.insert(std::make_pair(currLmName, currLm));
}


ImageLogger::~ImageLogger(void)
{
}

ImageLogger* ImageLogger::Instance(void)
{
	static ImageLogger instance;
	return &instance;
}

void ImageLogger::setVerboseLevelImages(int level)
{
	this->verboseLevelImages = level;
}


void ImageLogger::LogImgInputGray(const cv::Mat *img, std::string filename)
{
	if(this->global.img.writeImgInputGray) {
		size_t idx;
		idx = filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << filename.substr(idx+1) << "_0InputGray.png" << std::ends;
		cv::imwrite(fn.str(), *img);
	}
}

void ImageLogger::LogImgInputRGB(const cv::Mat *img, std::string filename)
{
	if(this->global.img.writeImgInputRGB) {
		size_t idx;
		idx = filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << filename.substr(idx+1) << "_0InputRGB.png" << std::ends;
		cv::imwrite(fn.str(), *img);
	}
}

void ImageLogger::LogImgPyramid(const Pyramid* pyr, std::string filename, int pyrIdx)
{
	if(this->global.img.writeImgPyramids) {
		size_t idx;
		idx = filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << filename.substr(idx+1) << "_Pyramid" << pyrIdx << "_w" << pyr->w << ".png" << std::ends;
		pyr->writePNG(fn.str());
	}
}

void ImageLogger::LogImgDetectorCandidates(const FdImage* img, std::vector<FdPatch*> candidates, std::string detectorIdForCertaintyColor, std::string filenameAppend)
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

		if(this->global.img.drawScales==true) {
			if(candidates.size()>0) {	// TODO this could be improved. I somehow need to reach the Detector.filtersize here... e.g. each Det attaches himself to the logger?
				this->drawAllScaleBoxes(outimg, &img->pyramids, detectorIdForCertaintyColor, candidates[0]->w, candidates[0]->h);
			} else {
				std::cout << "[Logger] Warning: Can't draw the scales because the candidates-list is empty." << std::endl;
			}
		}

		cv::imwrite(fn.str(), outimg);
	}
}

void ImageLogger::LogImgCircleDetectorCandidates( const FdImage* img, cv::vector<cv::Vec3f> circles, std::string detectorId)
{
	if((this->global.img.writeDetectorCandidates==true) || this->verboseLevelImages>=2) {
		/*if(filenameAppend=="") {
			filenameAppend = detectorIdForCertaintyColor;
		} else {
			filenameAppend = detectorIdForCertaintyColor + "_" + filenameAppend;
		}*/
/*
		std::string filenameAppend = detectorId;
		size_t idx;
		idx = img->filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << img->filename.substr(idx+1) << "_" << filenameAppend << ".png" << std::ends;

		cv::Mat outimg;
		outimg = img->data_matbgr.clone();	// leave input image intact
		
		for(size_t i = 0; i < circles.size(); i++) {
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// draw the circle center
			cv::circle( outimg, center, 2, cv::Scalar(0,255,0), -1, 8, 0 );
			// draw the circle outline
			cv::circle( outimg, center, radius, cv::Scalar(0,0,255), 2, 8, 0 );
		}
		cv::imwrite(fn.str(), outimg);
	}
}

void ImageLogger::LogImgDetectorProbabilityMap(const cv::Mat* probMap, const std::string filename, const std::string detectorId, std::string filenameAppend)
{
	if((this->global.img.writeDetectorProbabilityMaps==true) || this->verboseLevelImages>=2) {
		if(filenameAppend=="") {
			filenameAppend = detectorId;
		} else {
			filenameAppend = detectorId + "_" + filenameAppend;
		}
		size_t idx;
		idx = filename.find_last_of("/\\");
		std::ostringstream fn;
		fn << this->imgLogBasepath << filename.substr(idx+1) << "_" << filenameAppend << "_ProbabilityMap.png" << std::ends;

		cv::Mat outimg;
		outimg = probMap->clone();	// leave input image intact

		outimg *= 255.0;
		outimg.convertTo(outimg, CV_8U);
		
		cv::imwrite(fn.str(), outimg);
	}
}

void ImageLogger::LogImgDetectorFinal(const FdImage* img, std::vector<FdPatch*> candidates, std::string detectorIdForCertaintyColor, std::string filenameAppend)
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

void ImageLogger::LogImgDetectorFinal(const FdImage* img, std::vector<std::pair<std::string, std::vector<FdPatch*> > > candidates, std::string detectorIdForCertaintyColor, std::string filenameAppend)
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
		for(unsigned int i=0; i<candidates.size(); ++i) {
			this->drawFfpSymbols(outimg, candidates[i]);
		}

		cv::imwrite(fn.str(), outimg);
	}
}

void ImageLogger::LogImgRegressor(const FdImage* img, std::vector<FdPatch*> candidates, std::string regressorIdForFoutAngleColor, std::string filenameAppend)
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

void ImageLogger::LogImgRegressorPyramids(const FdImage* img, std::vector<FdPatch*> candidates, std::string regressorIdForFoutAngleColor, std::string filenameAppend)
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

void ImageLogger::drawBoxesWithCertainty(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor)
{
	std::vector<FdPatch*>::iterator pit = patches.begin();
	for(; pit != patches.end(); pit++) {
		cv::rectangle(rgbimg, cv::Point((*pit)->c.x-(*pit)->w_inFullImg/2, (*pit)->c.y-(*pit)->h_inFullImg/2), cv::Point((*pit)->c.x+(*pit)->w_inFullImg/2, (*pit)->c.y+(*pit)->h_inFullImg/2), cv::Scalar(0, 0, (float)255*(*pit)->certainty[identifierForColor]));
	}
}

void ImageLogger::drawBoxesWithAngleColor(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor)
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

void ImageLogger::drawCenterpointsWithAngleColor(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForColor, int scale)
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

void ImageLogger::drawFoutAsText(cv::Mat rgbimg, std::vector<FdPatch*> patches, std::string identifierForFout)
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

void ImageLogger::drawAllScaleBoxes(cv::Mat rgbimg, const PyramidMap* pyramids, std::string detectorId, int det_filtersize_x, int det_filtersize_y)
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

void ImageLogger::drawScaleBox(cv::Mat rgbimg, int w, int h)
{
	cv::Scalar color = cv::Scalar(175, 175, 175);	//cv::Scalar(b, g, r);
	cv::rectangle(rgbimg, cv::Point(0, 0), cv::Point(w-1, h-1), color);
}

void ImageLogger::drawFfpSymbols( cv::Mat image, std::pair<std::string, std::vector<FdPatch*> > candidates )
{
	for(unsigned int i=0; i<candidates.second.size(); ++i) {
		drawFfpSymbol(image, candidates.first, candidates.second[i]);
	}
}

void ImageLogger::drawFfpSymbol( cv::Mat image, std::string ffpName, FdPatch* patch )
{
	landmarkData thisLm = landmarksData.find(ffpName)->second;
	cv::Point2i realFfpCenter(patch->c.x+patch->w_inFullImg*thisLm.displacementFactorW, patch->c.y+patch->h_inFullImg*thisLm.displacementFactorH);
	unsigned int position = 0;
	for (int currRow = realFfpCenter.y-1; currRow<=realFfpCenter.y+1; ++currRow) {
		for (int currCol = realFfpCenter.x-1; currCol<=realFfpCenter.x+1; ++currCol) {
			if(thisLm.symbol[position]==true) {
				image.at<cv::Vec3b>(currRow,currCol)[0] = (uchar)cvRound(255.0f * thisLm.bgrColor.val[0]);
				image.at<cv::Vec3b>(currRow,currCol)[1] = (uchar)cvRound(255.0f * thisLm.bgrColor.val[1]);
				image.at<cv::Vec3b>(currRow,currCol)[2] = (uchar)cvRound(255.0f * thisLm.bgrColor.val[2]);
			}
			++position;
		}
	}

}

void ImageLogger::drawFfpsSmallSquare( cv::Mat img, std::vector<std::pair<std::string, cv::Point2f> > points )
{
	for (unsigned int i=0; i<points.size(); ++i) {
		drawFfpSmallSquare(img, points[i]);
	}
}

void ImageLogger::drawFfpSmallSquare( cv::Mat img, std::pair<std::string, cv::Point2f> point )
{
	cv::rectangle(img, cv::Point(cvRound(point.second.x-2.0f), cvRound(point.second.y-2.0f)), cv::Point(cvRound(point.second.x+2.0f), cvRound(point.second.y+2.0f)), cv::Scalar(255, 0, 0));
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

} /* namespace detection */

#include "stdafx.h"
#include "CircleDetector.h"

#include "SLogger.h"
#include "FdImage.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

CircleDetector::CircleDetector(void)
{
	this->identifier = "CircleDetector";
}


CircleDetector::~CircleDetector(void)
{
}


cv::vector<cv::Vec3f> CircleDetector::detectOnImage(FdImage* img)
{
	cv::Mat gray = img->data_matgray.clone();
	//cv::cvtColor(gray, gray, CV_BGR2GRAY);
    cv::GaussianBlur( gray, gray, cv::Size(9, 9), 2, 2 );
    cv::vector<cv::Vec3f> circles;		
	// good: 2 1 125 70 r/10 r/2
	// good: 1 1 90 20 r/8 r/3.5
	cv::HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1,  1,                 90,       20,                    gray.rows/8, gray.rows/3.5);
				/*	 in    out_vec  method             res minDistBetwCenters upperCanny howManyVotesToBeCircle minradius     maxradius	*/
	
	this->circles = circles;
	Logger->LogImgCircleDetectorCandidates(img, circles, this->identifier);
	
	return circles;
}

cv::Mat CircleDetector::getProbabilityMap(FdImage* img)
{
	cv::Mat probMap(img->data_matbgr.rows, img->data_matbgr.cols, CV_32FC1, cv::Scalar(0));

	for(size_t i = 0; i < circles.size(); i++) {
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		//cv::circle( outimg, center, 2, cv::Scalar(0,255,0), -1, 8, 0 );
		// draw the circle outline
		//cv::circle( outimg, center, radius, cv::Scalar(0,0,255), 2, 8, 0 );

		int gaussRange;
		if(radius % 2 == 0) {	// even
			gaussRange = radius+1;
		} else {	// odd
			gaussRange = radius;
		}

		cv::Mat kx = cv::getGaussianKernel(gaussRange, gaussRange/7.0, CV_32F);
		cv::Mat ky = cv::getGaussianKernel(gaussRange, gaussRange/7.0, CV_32F);
		cv::Mat kxy = kx*ky.t();

		float fac = kx.at<float>((gaussRange-1)/2);
		kx = kx * (1.0/fac);
		fac = ky.at<float>((gaussRange-1)/2);
		ky = ky * (1.0/fac);
		kxy = kx*ky.t();

		//matgray.at<uchar>(i, j); // (y, x) !!! i=row, j=column (matrix)
		// the filter
		/*cv::Mat out;
		cv::Mat tmp = kxy.clone();
		tmp *= 255.0;
		tmp.convertTo(out, CV_8U);
		std::ostringstream oss;
		oss << "out\\gaussfilter_circle_" << i << ".png";
		cv::imwrite(oss.str(), out);
		oss.str("");*/

		int skipLeft = 0; int skipRight = 0; int skipTop = 0; int skipBottom = 0;
		if(center.x-(gaussRange-1)/2 < 0)
			skipLeft = (gaussRange-1)/2-center.x;
		if(center.x+(gaussRange-1)/2 > img->data_matbgr.cols-1)
			skipRight = (gaussRange-1)/2-((img->data_matbgr.cols-1)-center.x);
		if(center.y-(gaussRange-1)/2 < 0)
			skipTop = (gaussRange-1)/2-center.y;
		if(center.y+(gaussRange-1)/2 > img->data_matbgr.rows-1)
			skipBottom = (gaussRange-1)/2-((img->data_matbgr.rows-1)-center.y);

		int ikx=0; int iky=0;
		for(int x=(center.x-(gaussRange-1)/2)+skipLeft; x<=(center.x+(gaussRange-1)/2)-skipRight; ++x) {
			for(int y=(center.y-(gaussRange-1)/2)+skipTop; y<=(center.y+(gaussRange-1)/2)-skipBottom; ++y) {
				//x, y = pos in probMap
				if(probMap.at<float>(y, x) < kxy.at<float>(ikx, iky))	// if the new value is bigger, replace it. Else keep old one.
					probMap.at<float>(y, x) = kxy.at<float>(ikx, iky);
				iky++;
			}
			iky=0;
			ikx++;
		}

		/* Debug output: Each individual probability map. */
		/*cv::Mat out2;
		cv::Mat tmp2 = probMap.clone();
		tmp2 *= 255.0;
		tmp2.convertTo(out2, CV_8U);
		std::ostringstream oss2;
		oss2 << "out\\probMap" << i << "b.png";
		cv::imwrite(oss2.str(), out2);
		oss2.str("");*/

	}
	Logger->LogImgDetectorProbabilityMap(&probMap, img->filename, this->getIdentifier());
	return probMap;

}
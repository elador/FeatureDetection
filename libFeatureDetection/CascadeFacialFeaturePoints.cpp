#include "stdafx.h"
#include "CascadeFacialFeaturePoints.h"
#include "CascadeFacialFeaturePointsSimple.h"
#include "CircleDetector.h"
#include "SkinDetector.h"

#include "SLogger.h"
#include "FdPatch.h"
#include "FdImage.h"

CascadeFacialFeaturePoints::CascadeFacialFeaturePoints(void)
{
	wvm_frontal = new DetectorWVM();
	wvm_frontal->load("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_fd.mat");

	oe = new OverlapElimination();
	oe->load("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_fd.mat");

	circleDet = new CircleDetector();
	skinDet = new SkinDetector();

	ffpCasc = new CascadeFacialFeaturePointsSimple();
	ffpCasc->setIdentifier("5ffpCascade");
}


CascadeFacialFeaturePoints::~CascadeFacialFeaturePoints(void)
{
	delete wvm_frontal;
	delete oe;

	delete circleDet;
	delete skinDet;

	delete ffpCasc; 
}


int CascadeFacialFeaturePoints::initForImage(FdImage* myimg)
{
	//face_frontal->initForImage(myimg);
	wvm_frontal->initForImage(myimg);
	return 1;
}

int CascadeFacialFeaturePoints::detectOnImage(FdImage* myimg)
{
	wvm_frontal->extractToPyramids(myimg);
	this->candidates = wvm_frontal->detectOnImage(myimg);
	Logger->LogImgDetectorCandidates(myimg, candidates, wvm_frontal->getIdentifier(), "1WVM");
	std::vector<cv::Mat> probabilityMapsWVM = wvm_frontal->getProbabilityMaps(myimg);

	std::vector<FdPatch*> afterFirstOE;
	//0: only after SVM, 1: only before SVM, n: Reduce each cluster to n before SVM and to 1 after; Default: 1
	if(oe->doOE!=0) {		// do overlapeliminiation after RVM before SVM
		if (oe->doOE==1)
			afterFirstOE = oe->eliminate(this->candidates, wvm_frontal->getIdentifier());
		else
			//oe->eliminate(faces, args.pp_oe_percent, args.doesPPOverlapElimination, false);	// reduce to "doOE"-num Clusters
			afterFirstOE = oe->eliminate(this->candidates, wvm_frontal->getIdentifier());
			
	} else {
		afterFirstOE = this->candidates;
	}
	Logger->LogImgDetectorCandidates(myimg, afterFirstOE, wvm_frontal->getIdentifier(), "2WVMoe");

	// LOST Logger->LogImgCanny(&myimg->data_matbgr, myimg->filename);

	circleDet->detectOnImage(myimg);
	cv::Mat circleProbMap = circleDet->getProbabilityMap(myimg);

	cv::Mat binarySkinMap = skinDet->detectOnImage(myimg);

	//cv::Mat final = 0.5*probabilityMapsWVM[1] + 0.3*binarySkinMap + 0.2*circleProbMap;
	//Logger->LogImgDetectorProbabilityMap(&final, myimg->filename, "FINAL");
	
	ffpCasc->initForImage(myimg);	// create the pyramids (and set ROI to default)

	std::sort(afterFirstOE.begin(), afterFirstOE.end(), FdPatch::SortByCertainty(wvm_frontal->getIdentifier()));	// only necessary if we didnt run an OE?

	std::vector<FdPatch*>::iterator itr;
	for (itr = afterFirstOE.begin(); itr != afterFirstOE.end(); ++itr ) {
		/* The region of the FD patch. For FFP-Det: */
		int roiLeft = (*itr)->c.x-(*itr)->w_inFullImg/2;
		int roiRight = (*itr)->c.x+(*itr)->w_inFullImg/2;
		int roiTop = (*itr)->c.y-(*itr)->h_inFullImg/2;
		int roiBottom = (*itr)->c.y+(*itr)->h_inFullImg/2;
		Rect roi(roiLeft, roiTop, roiRight, roiBottom);
		
		cv::Mat test = myimg->data_matbgr.clone();
		cv::rectangle(test, cv::Point(roi.left, roi.top), cv::Point(roi.right, roi.bottom), cv::Scalar(0, 0, 255));
		imwrite("out\\asfd2.png", test);/*
		cv::Mat temp = myimg->data_matbgr(cv::Rect(roiLeft, roiTop, (*itr)->w_inFullImg, (*itr)->h_inFullImg));
		imwrite("out\\asdf.png", temp);*/
		ffpCasc->setRoiInImage(roi);
		ffpCasc->detectOnImage(myimg);
		std::vector<cv::Mat> probabilityMapsWVMre = ffpCasc->reye->wvm->getProbabilityMaps(myimg);
		std::vector<cv::Mat> probabilityMapsWVMle = ffpCasc->leye->wvm->getProbabilityMaps(myimg);
		std::vector<cv::Mat> probabilityMapsWVMnt = ffpCasc->nosetip->wvm->getProbabilityMaps(myimg);
		std::vector<cv::Mat> probabilityMapsWVMrm = ffpCasc->rmouth->wvm->getProbabilityMaps(myimg);
		std::vector<cv::Mat> probabilityMapsWVMlm = ffpCasc->lmouth->wvm->getProbabilityMaps(myimg);
		
	}

	return 1;
}


/* A test:
cv::Mat blurred = myimg->data_matbgr.clone();
cv::cvtColor(blurred, blurred, CV_BGR2GRAY);
cv::blur(blurred, blurred, cv::Size(3, 3));
cv::Mat edges;
cv::Canny(blurred, edges, 30, 90);

cv::Mat tmp(edges.rows, edges.cols, CV_8U, cv::Scalar(255));
cv::Mat cannyInv = tmp - edges;
cv::imwrite("out\\ci.png", cannyInv);
cv::Mat skinInv = tmp - binarySkinMap;
cv::imwrite("out\\si.png", skinInv);
cv::Mat final = skinInv + cannyInv;

cv::imwrite("out\\test.png", final);	// == edges around/in the skin region!
*/
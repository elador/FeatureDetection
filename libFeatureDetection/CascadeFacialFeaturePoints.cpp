#include "StdAfx.h"
#include "CascadeFacialFeaturePoints.h"


CascadeFacialFeaturePoints::CascadeFacialFeaturePoints(void)
{
	//face_frontal = new CascadeWvmOeSvmOe("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat");
	//eye_left = new CascadeWvmOeSvmOe("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat");

	wvm_frontal = new DetectorWVM();
	wvm_frontal->load("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat");

	oe = new OverlapElimination();
	oe->load("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat");

	circleDet = new CircleDetector();
	skinDet = new SkinDetector();

	ffpCasc = new CascadeFacialFeaturePointsSimple();
}


CascadeFacialFeaturePoints::~CascadeFacialFeaturePoints(void)
{
	//delete face_frontal;
	//delete eye_left;
	delete wvm_frontal;
	delete oe;

	delete circleDet;
	delete skinDet;

	delete ffpCasc; 
}


int CascadeFacialFeaturePoints::init_for_image(FdImage* myimg)
{
	//face_frontal->init_for_image(myimg);
	wvm_frontal->init_for_image(myimg);
	return 1;
}

int CascadeFacialFeaturePoints::detect_on_image(FdImage* myimg)
{
	wvm_frontal->extractToPyramids(myimg);
	this->candidates = wvm_frontal->detect_on_image(myimg);
	// LOST std::vector<cv::Mat> probabilityMapsWVM = wvm_frontal->getProbabilityMaps(myimg);
	Logger->LogImgDetectorCandidates(myimg, candidates, wvm_frontal->getIdentifier(), "1WVM");
	
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
	
	ffpCasc->initForImage(myimg);

	std::vector<FdPatch*>::iterator itr;
	for (itr = afterFirstOE.begin(); itr != afterFirstOE.end(); ++itr ) {
		/* The region of the FD patch. For FFP-Det: */
		int roiLeft = (*itr)->c.x-(*itr)->w_inFullImg/2;
		int roiRight = (*itr)->c.x+(*itr)->w_inFullImg/2;
		int roiTop = (*itr)->c.y-(*itr)->h_inFullImg/2;
		int roiBottom = (*itr)->c.y+(*itr)->h_inFullImg/2;
		ffpCasc->reye->wvm->roi_inImg = Rect(roiLeft, roiTop, roiRight, roiBottom);
		ffpCasc->reye->svm->roi_inImg = Rect(roiLeft, roiTop, roiRight, roiBottom);
		ffpCasc->detectOnImage(myimg);
		
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
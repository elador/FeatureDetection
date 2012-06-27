#include "stdafx.h"
#include "CascadeERT.h"


CascadeERT::CascadeERT(void)
{
	wvm = new DetectorWVM();	// only init and set everything to 0
	svm = new DetectorSVM();	// only init and set everything to 0
	
	wvmr = new DetectorWVM();
	svmr = new DetectorSVM();

	wvml = new DetectorWVM();
	svml = new DetectorSVM();

	svr = new RegressorSVR();
	wvr = new RegressorWVR();

	oe = new OverlapElimination();
	oer = new OverlapElimination();
	oel = new OverlapElimination();
}


CascadeERT::~CascadeERT(void)
{
	delete wvm;
	delete svm;
	delete wvmr;
	delete svmr;
	delete wvml;
	delete svml;
	delete svr;
	delete wvr;
	delete oe;
	delete oer;
	delete oel;
}


int CascadeERT::init_for_image(FdImage* myimg)
{

	wvm->setIdentifier("1_WVM_Center");
	wvmr->setIdentifier("1_WVM_Right");
	wvml->setIdentifier("1_WVM_Left");
	wvr->setIdentifier("3_WVR");
	svr->setIdentifier("5_SVR");
	svm->setIdentifier("4_DetectorSVM_Center");
	svmr->setIdentifier("4_DetectorSVM_Right");
	svml->setIdentifier("4_DetectorSVM_Left");
	oe->setIdentifier("2_OE_Center");
	oer->setIdentifier("2_OE_Right");
	oel->setIdentifier("2_OE_Left");

	this->candidates.clear();

	wvr->init_for_image(myimg);
	wvm->init_for_image(myimg);	// Theoretically, only the first would need to create the pyramid. 
	svm->init_for_image(myimg);	// Because in this case, all other detectors work on a patchlist.
	svr->init_for_image(myimg);	// Well, NO, because each detector has to register himself into the pyramids! But this doesnt cost time.
	
	wvmr->init_for_image(myimg);
	svmr->init_for_image(myimg);
	wvml->init_for_image(myimg);
	svml->init_for_image(myimg);
	return 1;
}

int CascadeERT::detect_on_image(FdImage* myimg)
{

	//wvm->extract(myimg);
	//candidates = svr->detect_on_image(myimg);
	//Logger->LogImgRegressorPyramids(myimg, candidates, svr->getIdentifier());
	//Logger->LogImgRegressor(myimg, candidates, wvr->getIdentifier());

	//svr->detect_on_patchvec(candidates);
	//Logger->LogImgRegressorPyramids(myimg, candidates, svr->getIdentifier());

	std::vector<FdPatch*> candidates_c;
	std::vector<FdPatch*> candidates_l;
	std::vector<FdPatch*> candidates_r;

	wvm->extractToPyramids(myimg);
	
	/* Step 1: All 3 WVMs */
	candidates_c = wvm->detect_on_image(myimg);
	candidates_r = wvmr->detect_on_image(myimg);
	candidates_l = wvml->detect_on_image(myimg);

	/* Step 2: OE */
	candidates_c = oe->eliminate(candidates_c, wvm->getIdentifier());
	candidates_r = oer->eliminate(candidates_r, wvmr->getIdentifier());
	candidates_l = oel->eliminate(candidates_l, wvml->getIdentifier());

	Logger->LogImgDetectorCandidates(myimg, candidates_c, wvm->getIdentifier(), "afterOE");
	Logger->LogImgDetectorCandidates(myimg, candidates_r, wvmr->getIdentifier(), "afterOE");
	Logger->LogImgDetectorCandidates(myimg, candidates_l, wvml->getIdentifier(), "afterOE");

	/* Step 3: WVR on ROI around each candidate */
	/* FOR CENTER WVM PATCHES */
	int clusterCounter = 0;
	for(std::vector<FdPatch*>::iterator itc = candidates_c.begin(); itc != candidates_c.end(); ++itc) {
		// For each candidate after OE:
		// extract ROI
		std::vector<FdPatch*> theOneCandidateInCenter;
		theOneCandidateInCenter.push_back(*itc);
		std::vector<FdPatch*> cand_roi_c = wvr->getPatchesROI(myimg, (*itc)->c.x_py, (*itc)->c.y_py, (*itc)->c.s, 1, 1, 0, wvr->getIdentifier());
		Logger->LogImgRegressor(myimg, theOneCandidateInCenter, wvr->getIdentifier(), "MIDDLE");
		wvr->detect_on_patchvec(cand_roi_c);
		std::ostringstream clusterName;
		clusterName << "c_cluster" << clusterCounter;
		Logger->LogImgRegressor(myimg, cand_roi_c, wvr->getIdentifier(), clusterName.str());
		// get avg-angle
		float avg_angle_cand_roi_c = 0.0f;
		for(std::vector<FdPatch*>::iterator itc = cand_roi_c.begin(); itc != cand_roi_c.end(); ++itc) {
			avg_angle_cand_roi_c += (*itc)->fout[wvr->getIdentifier()];
		}
		avg_angle_cand_roi_c /= (float)cand_roi_c.size();
		
		
		// run respective SVM
		if(avg_angle_cand_roi_c < -35) {
			//-90 = LOOKING RIGHT
			theOneCandidateInCenter = svmr->detect_on_patchvec(theOneCandidateInCenter);
		} else if(avg_angle_cand_roi_c > 35) {
			//+90 = LOOKING LEFT
			theOneCandidateInCenter = svml->detect_on_patchvec(theOneCandidateInCenter);
		} else {
			//middle
			theOneCandidateInCenter = svm->detect_on_patchvec(theOneCandidateInCenter);
		}// MR: Alle SVM laufen lassen
		// ExpNumFP (not necessary)
		// Is it a face? If yes
		if(theOneCandidateInCenter.size()>0) {
			// SVR around ROI
			cand_roi_c = svr->getPatchesROI(myimg, theOneCandidateInCenter[0]->c.x_py, theOneCandidateInCenter[0]->c.y_py, theOneCandidateInCenter[0]->c.s, 1, 1, 1, svr->getIdentifier());
			//svr->detect_on_patchvec(cand_roi_c);
			std::ostringstream roiName;
			roiName << "c_roi" << clusterCounter;
			Logger->LogImgRegressor(myimg, cand_roi_c, svr->getIdentifier(), roiName.str());
			float avg_angle_cand_roi_c = 0.0f;
			for(std::vector<FdPatch*>::iterator itc = cand_roi_c.begin(); itc != cand_roi_c.end(); ++itc) {
				avg_angle_cand_roi_c += (*itc)->fout[svr->getIdentifier()];
			}
			avg_angle_cand_roi_c /= (float)cand_roi_c.size();
			// -> Final box+angle
			std::ostringstream finalAngle;
			finalAngle << "c_cluster" << clusterCounter << "_FINAL_yawavg_" << avg_angle_cand_roi_c;
			Logger->LogImgRegressor(myimg, theOneCandidateInCenter, svr->getIdentifier(), finalAngle.str());
			this->candidates.push_back(theOneCandidateInCenter[0]);
		}
		// if no, continue loop with next candidate.

		++clusterCounter;
	}


	/* FOR LEFT WVM PATCHES */
	clusterCounter = 0;
	for(std::vector<FdPatch*>::iterator itc = candidates_l.begin(); itc != candidates_l.end(); ++itc) {
		// For each candidate after OE:
		// extract ROI
		std::vector<FdPatch*> theOneCandidateInCenter;
		theOneCandidateInCenter.push_back(*itc);
		std::vector<FdPatch*> cand_roi_l = wvr->getPatchesROI(myimg, (*itc)->c.x_py, (*itc)->c.y_py, (*itc)->c.s, 1, 1, 0, wvr->getIdentifier());
		wvr->detect_on_patchvec(cand_roi_l);
		std::ostringstream clusterName;
		clusterName << "l_cluster" << clusterCounter;
		Logger->LogImgRegressor(myimg, cand_roi_l, wvr->getIdentifier(), clusterName.str());
		// get avg-angle
		float avg_angle_cand_roi_l = 0.0f;
		for(std::vector<FdPatch*>::iterator itc = cand_roi_l.begin(); itc != cand_roi_l.end(); ++itc) {
			avg_angle_cand_roi_l += (*itc)->fout[wvr->getIdentifier()];
		}
		avg_angle_cand_roi_l /= (float)cand_roi_l.size();
		
		
		// run respective SVM
		if(avg_angle_cand_roi_l < -35) {
			//-90 = LOOKING RIGHT
			theOneCandidateInCenter = svmr->detect_on_patchvec(theOneCandidateInCenter);
		} else if(avg_angle_cand_roi_l > 35) {
			//+90 = LOOKING LEFT
			theOneCandidateInCenter = svml->detect_on_patchvec(theOneCandidateInCenter);
		} else {
			//middle
			theOneCandidateInCenter = svm->detect_on_patchvec(theOneCandidateInCenter);
		}
		// ExpNumFP (not necessary)
		// Is it a face? If yes
		if(theOneCandidateInCenter.size()>0) {
			// SVR around ROI
			cand_roi_l = svr->getPatchesROI(myimg, theOneCandidateInCenter[0]->c.x_py, theOneCandidateInCenter[0]->c.y_py, theOneCandidateInCenter[0]->c.s, 1, 1, 1, svr->getIdentifier());
			//svr->detect_on_patchvec(cand_roi_l);
			std::ostringstream roiName;
			roiName << "l_roi" << clusterCounter;
			Logger->LogImgRegressor(myimg, cand_roi_l, svr->getIdentifier(), roiName.str());
			float avg_angle_cand_roi_l = 0.0f;
			for(std::vector<FdPatch*>::iterator itc = cand_roi_l.begin(); itc != cand_roi_l.end(); ++itc) {
				avg_angle_cand_roi_l += (*itc)->fout[svr->getIdentifier()];
			}
			avg_angle_cand_roi_l /= (float)cand_roi_l.size();
			// -> Final box+angle
			std::ostringstream finalAngle;
			finalAngle << "l_cluster" << clusterCounter << "_FINAL_yawavg_" << avg_angle_cand_roi_l;
			Logger->LogImgRegressor(myimg, theOneCandidateInCenter, svr->getIdentifier(), finalAngle.str());
			this->candidates.push_back(theOneCandidateInCenter[0]);
		}
		// if no, continue loop with next candidate.

		++clusterCounter;
	}




	/* FOR RIGHT WVM PATCHES */
	clusterCounter = 0;
	for(std::vector<FdPatch*>::iterator itc = candidates_r.begin(); itc != candidates_r.end(); ++itc) {
		// For each candidate after OE:
		// extract ROI
		std::vector<FdPatch*> theOneCandidateInCenter;
		theOneCandidateInCenter.push_back(*itc);
		std::vector<FdPatch*> cand_roi_r = wvr->getPatchesROI(myimg, (*itc)->c.x_py, (*itc)->c.y_py, (*itc)->c.s, 1, 1, 0, wvr->getIdentifier());
		wvr->detect_on_patchvec(cand_roi_r);
		std::ostringstream clusterName;
		clusterName << "r_cluster" << clusterCounter;
		Logger->LogImgRegressor(myimg, cand_roi_r, wvr->getIdentifier(), clusterName.str());
		// get avg-angle
		float avg_angle_cand_roi_r = 0.0f;
		for(std::vector<FdPatch*>::iterator itc = cand_roi_r.begin(); itc != cand_roi_r.end(); ++itc) {
			avg_angle_cand_roi_r += (*itc)->fout[wvr->getIdentifier()];
		}
		avg_angle_cand_roi_r /= (float)cand_roi_r.size();
		
		
		// run respective SVM
		if(avg_angle_cand_roi_r < -35) {
			//-90 = LOOKING RIGHT
			theOneCandidateInCenter = svmr->detect_on_patchvec(theOneCandidateInCenter);
		} else if(avg_angle_cand_roi_r > 35) {
			//+90 = LOOKING LEFT
			theOneCandidateInCenter = svml->detect_on_patchvec(theOneCandidateInCenter);
		} else {
			//middle
			theOneCandidateInCenter = svm->detect_on_patchvec(theOneCandidateInCenter);
		}
		// ExpNumFP (not necessary)
		// Is it a face? If yes
		if(theOneCandidateInCenter.size()>0) {
			// SVR around ROI
			cand_roi_r = svr->getPatchesROI(myimg, theOneCandidateInCenter[0]->c.x_py, theOneCandidateInCenter[0]->c.y_py, theOneCandidateInCenter[0]->c.s, 1, 1, 1, svr->getIdentifier());
			//svr->detect_on_patchvec(cand_roi_r);
			std::ostringstream roiName;
			roiName << "r_roi" << clusterCounter;
			Logger->LogImgRegressor(myimg, cand_roi_r, svr->getIdentifier(), roiName.str());
			float avg_angle_cand_roi_r = 0.0f;
			for(std::vector<FdPatch*>::iterator itc = cand_roi_r.begin(); itc != cand_roi_r.end(); ++itc) {
				avg_angle_cand_roi_r += (*itc)->fout[svr->getIdentifier()];
			}
			avg_angle_cand_roi_r /= (float)cand_roi_r.size();
			// -> Final box+angle
			std::ostringstream finalAngle;
			finalAngle << "r_cluster" << clusterCounter << "_FINAL_yawavg_" << avg_angle_cand_roi_r;
			Logger->LogImgRegressor(myimg, theOneCandidateInCenter, svr->getIdentifier(), finalAngle.str());
			this->candidates.push_back(theOneCandidateInCenter[0]);
		}
		// if no, continue loop with next candidate.

		++clusterCounter;
	}


	/*std::sort(this->candidates.begin(), this->candidates.end(), FdPatch::SortByCertainty(svm->getIdentifier())); // "hack", should also consider l/r svm
	int start=0;
	for(std::vector<FdPatch*>::iterator itc = candidates.begin(); itc != candidates.end();) {
		if(start==0) {
			++itc;
		} else {

			itc = this->candidates.erase(itc);
		}
	}*/


	/*candidates_c = oe->eliminate(candidates_c, wvm->getIdentifier());
	candidates_r = oer->eliminate(candidates_r, wvmr->getIdentifier());
	candidates_l = oel->eliminate(candidates_l, wvml->getIdentifier());
	Logger->LogImgDetectorCandidates(myimg, candidates_c, wvm->getIdentifier(), oe->getIdentifier());
	Logger->LogImgDetectorCandidates(myimg, candidates_r, wvmr->getIdentifier(), oer->getIdentifier());
	Logger->LogImgDetectorCandidates(myimg, candidates_l, wvml->getIdentifier(), oel->getIdentifier());

	candidates_c = svm->detect_on_patchvec(candidates_c);
	candidates_r = svmr->detect_on_patchvec(candidates_r);
	candidates_l = svml->detect_on_patchvec(candidates_l);
	Logger->LogImgDetectorCandidates(myimg, candidates_c, svm->getIdentifier());
	Logger->LogImgDetectorCandidates(myimg, candidates_r, svmr->getIdentifier());
	Logger->LogImgDetectorCandidates(myimg, candidates_l, svml->getIdentifier());

	candidates_c = oe->exp_num_fp_elimination(candidates_c, svm->getIdentifier());
	candidates_r = oer->exp_num_fp_elimination(candidates_r, svmr->getIdentifier());
	candidates_l = oel->exp_num_fp_elimination(candidates_l, svml->getIdentifier());
	Logger->LogImgDetectorCandidates(myimg, candidates_c, svm->getIdentifier(), "OE2_Center");
	Logger->LogImgDetectorCandidates(myimg, candidates_r, svmr->getIdentifier(), "OE2_Right");
	Logger->LogImgDetectorCandidates(myimg, candidates_l, svml->getIdentifier(), "OE2_Left");



	*/
	/*wvr->extract(myimg);
	std::vector<FdPatch*> candidates_wvr;
	std::cout << "[CascadeERT] Running WVR on whole image" << std::endl;
	candidates_wvr = wvr->detect_on_image(myimg);
	*/


	/*std::vector<FdPatch*> candidates_wvr_center;
	std::vector<FdPatch*> candidates_wvr_left;
	std::vector<FdPatch*> candidates_wvr_right;

	int left=0; int right=0; int center=0;

	std::vector<FdPatch*>::iterator itr;
	for (itr = candidates_wvr.begin(); itr != candidates_wvr.end(); ++itr ) {
		if((*itr)->fout[wvr->getIdentifier()] < -35) {
			//-90 = LOOKING RIGHT
			candidates_wvr_right.push_back((*itr));
			right++;
		} else if((*itr)->fout[wvr->getIdentifier()] > 35) {
			//+90 = LOOKING LEFT
			candidates_wvr_left.push_back((*itr));
			left++;
		} else {
			//middle
			candidates_wvr_center.push_back((*itr));
			center++;
		}
	}

	std::vector<FdPatch*> candidates_wvm_center;
	std::cout << "[CascadeERT] Running Center-WVM on center-patches" << std::endl;
	candidates_wvm_center = wvm->detect_on_patchvec(candidates_wvr_center);

	std::vector<FdPatch*> candidates_svr_center;
	candidates_svr_center = svr->detect_on_patchvec(candidates_wvm_center);
	
	std::vector<FdPatch*> candidates_svm_center;
	candidates_svm_center = svm->detect_on_patchvec(candidates_svr_center);

	this->candidates = candidates_svm_center;
	//tmp.clear();

	*/

	//Logger->drawBoxes(myimg->data_matbgr, candidates_svm_center);
	//Logger->drawYawAngle(myimg->data_matbgr, candidates_svm_center);

	// Debug output: Output all patches that passed the SVM, write their SVR value in the filename!
	/*std::vector<FdPatch*>::iterator itr2;
	int i=0;
	char* bla = new char[500];
	for (itr2 = candidates.begin(); itr2 != candidates.end(); ++itr2 ) {
		if((*itr2)->certainty[svm->getIdentifier()]>=0.90) {
			sprintf(bla, "face_yawout%1.3f.png", (*itr2)->fout[svr->getIdentifier()]);
			(*itr2)->writePNG(bla);
			i++;
		}
	}
	delete[] bla;*/

	//Logger->drawBoxesWithYawAngleColor(myimg->
	
	return 1;
}


/*
void CascadeERT::scaleDown(std::vector<FdPatch*> patches)
{

	std::vector<FdPatch*>::iterator itr2;
	for (itr2 = patches.begin(); itr2 != patches.end(); ++itr2 ) {
		
		cv::Mat test((*itr2)->h, (*itr2)->w, CV_8UC1, (*itr2)->data);
		cv::Mat dest;
		cv::resize(test, dest, cv::Size(20, 20), 0, 0, 1);
		
		(*itr2)->w = dest.cols;
		(*itr2)->h = dest.rows;
		//delete (*itr2)->iimg_x; 
		(*itr2)->iimg_x = NULL;
		//delete (*itr2)->iimg_xx; 
		(*itr2)->iimg_xx = NULL;
		//delete[] (*itr2)->data;
		(*itr2)->data = NULL;	// original data lives on. It is owned by the pyramid.
		(*itr2)->data = new unsigned char[(*itr2)->w*(*itr2)->h];

		for (int i=0; i<dest.rows; i++)
		{
			for (int j=0; j<dest.cols; j++)
			{
				(*itr2)->data[i*(*itr2)->w+j] = dest.at<uchar>(i, j); // (y, x) !!! i=row, j=column (matrix)
			}
		}
		//(*itr2)->writePNG();
	}



}
void CascadeERT::scaleUp(std::vector<FdPatch*> patches_to_scale_up, std::vector<FdPatch*> patchlist_original_size)
{
	std::vector<FdPatch*>::iterator itr2;
	for (itr2 = patches_to_scale_up.begin(); itr2 != patches_to_scale_up.end(); ++itr2 ) {

		delete (*itr2)->iimg_x; (*itr2)->iimg_x = NULL;
		delete (*itr2)->iimg_xx; (*itr2)->iimg_xx = NULL;
		delete[] (*itr2)->data;
		
		std::vector<FdPatch*>::iterator f;
		bool found=false;
		for (f = patchlist_original_size.begin(); f != patchlist_original_size.end(); ++f ) {
			if((*f)->c.s==(*itr2)->c.s && (*f)->c.x_py==(*itr2)->c.x_py && (*f)->c.y_py==(*itr2)->c.y_py) {
				if(found==true)
					std::cout << "Should not happen...2x" << std::endl;
				(*itr2)->data = (*f)->data;
				(*itr2)->w = 32;
				(*itr2)->h = 32;
				found=true;
			}
		}
		if(found==false)
			std::cout << "Should not happen... not found in list" << std::endl;
		

		//(*itr2)->writePNG();
	}

}
*/
/* Scale up/down ghetto... I think some pointers wrong...
int CascadeERT::detect_on_image(FdImage* myimg)
{
	//wvm->extract(myimg);
	wvr->extract(myimg);
	std::vector<FdPatch*> candidates_wvr32;
	candidates_wvr32 = wvr->detect_on_image(myimg);

	std::vector<FdPatch*> candidates_wvr_center32;

	int left=0; int right=0; int center=0;

	std::vector<FdPatch*>::iterator itr;
	for (itr = candidates_wvr32.begin(); itr != candidates_wvr32.end(); ++itr ) {
		if((*itr)->fout < -35) {
			//l/r
			left++;
		} else if((*itr)->fout > 35) {
			//l/r
			right++;
		} else {
			//middle
			candidates_wvr_center32.push_back((*itr));
			center++;
		}
	}

	scaleDown(candidates_wvr_center32); // now 20

	std::vector<FdPatch*> candidates_wvm_center20;
	candidates_wvm_center20 = wvm->detect_on_patchvec(candidates_wvr_center32); //20

	std::vector<FdPatch*> candidates_svr_center32;
	scaleUp(candidates_wvm_center20, candidates_wvr32);	// 32
	candidates_svr_center32 = svr->detect_on_patchvec(candidates_wvm_center20);	//32
	
	scaleDown(candidates_svr_center32);	// 20
	std::vector<FdPatch*> candidates_svm_center20;
	candidates_svm_center20 = svm->detect_on_patchvec(candidates_svr_center32);	//20

	this->candidates = candidates_svm_center20;
	//tmp.clear();

	return 1;
}
*/
int CascadeERT::detectOnImage(FdImage* myimg)
{

	//wvm->extract(myimg);
	//candidates = svr->detectOnImage(myimg);
	//Logger->LogImgRegressorPyramids(myimg, candidates, svr->getIdentifier());
	//Logger->LogImgRegressor(myimg, candidates, wvr->getIdentifier());

	//svr->detectOnPatchvec(candidates);
	//Logger->LogImgRegressorPyramids(myimg, candidates, svr->getIdentifier());

	std::vector<FdPatch*> candidates_c;
	std::vector<FdPatch*> candidates_l;
	std::vector<FdPatch*> candidates_r;

	wvm->extractToPyramids(myimg);
	
	/* Step 1: All 3 WVMs */
	candidates_c = wvm->detectOnImage(myimg);
	candidates_r = wvmr->detectOnImage(myimg);
	candidates_l = wvml->detectOnImage(myimg);

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
		wvr->detectOnPatchvec(cand_roi_c);
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
			theOneCandidateInCenter = svmr->detectOnPatchvec(theOneCandidateInCenter);
		} else if(avg_angle_cand_roi_c > 35) {
			//+90 = LOOKING LEFT
			theOneCandidateInCenter = svml->detectOnPatchvec(theOneCandidateInCenter);
		} else {
			//middle
			theOneCandidateInCenter = svm->detectOnPatchvec(theOneCandidateInCenter);
		}// MR: Alle SVM laufen lassen
		// ExpNumFP (not necessary)
		// Is it a face? If yes
		if(theOneCandidateInCenter.size()>0) {
			// SVR around ROI
			cand_roi_c = svr->getPatchesROI(myimg, theOneCandidateInCenter[0]->c.x_py, theOneCandidateInCenter[0]->c.y_py, theOneCandidateInCenter[0]->c.s, 1, 1, 1, svr->getIdentifier());
			//svr->detectOnPatchvec(cand_roi_c);
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
		wvr->detectOnPatchvec(cand_roi_l);
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
			theOneCandidateInCenter = svmr->detectOnPatchvec(theOneCandidateInCenter);
		} else if(avg_angle_cand_roi_l > 35) {
			//+90 = LOOKING LEFT
			theOneCandidateInCenter = svml->detectOnPatchvec(theOneCandidateInCenter);
		} else {
			//middle
			theOneCandidateInCenter = svm->detectOnPatchvec(theOneCandidateInCenter);
		}
		// ExpNumFP (not necessary)
		// Is it a face? If yes
		if(theOneCandidateInCenter.size()>0) {
			// SVR around ROI
			cand_roi_l = svr->getPatchesROI(myimg, theOneCandidateInCenter[0]->c.x_py, theOneCandidateInCenter[0]->c.y_py, theOneCandidateInCenter[0]->c.s, 1, 1, 1, svr->getIdentifier());
			//svr->detectOnPatchvec(cand_roi_l);
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
		wvr->detectOnPatchvec(cand_roi_r);
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
			theOneCandidateInCenter = svmr->detectOnPatchvec(theOneCandidateInCenter);
		} else if(avg_angle_cand_roi_r > 35) {
			//+90 = LOOKING LEFT
			theOneCandidateInCenter = svml->detectOnPatchvec(theOneCandidateInCenter);
		} else {
			//middle
			theOneCandidateInCenter = svm->detectOnPatchvec(theOneCandidateInCenter);
		}
		// ExpNumFP (not necessary)
		// Is it a face? If yes
		if(theOneCandidateInCenter.size()>0) {
			// SVR around ROI
			cand_roi_r = svr->getPatchesROI(myimg, theOneCandidateInCenter[0]->c.x_py, theOneCandidateInCenter[0]->c.y_py, theOneCandidateInCenter[0]->c.s, 1, 1, 1, svr->getIdentifier());
			//svr->detectOnPatchvec(cand_roi_r);
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

	candidates_c = svm->detectOnPatchvec(candidates_c);
	candidates_r = svmr->detectOnPatchvec(candidates_r);
	candidates_l = svml->detectOnPatchvec(candidates_l);
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
	candidates_wvr = wvr->detectOnImage(myimg);
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
	candidates_wvm_center = wvm->detectOnPatchvec(candidates_wvr_center);

	std::vector<FdPatch*> candidates_svr_center;
	candidates_svr_center = svr->detectOnPatchvec(candidates_wvm_center);
	
	std::vector<FdPatch*> candidates_svm_center;
	candidates_svm_center = svm->detectOnPatchvec(candidates_svr_center);

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

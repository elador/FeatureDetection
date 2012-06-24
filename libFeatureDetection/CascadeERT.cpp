#include "StdAfx.h"
#include "CascadeERT.h"


CascadeERT::CascadeERT(void)
{
	wvm = new DetectorWVM();	// only init and set everything to 0
	svm = new DetectorSVM();	// only init and set everything to 0
	svr = new RegressorSVR();
	wvr = new RegressorWVR();
}


CascadeERT::~CascadeERT(void)
{
	delete wvm;
	delete svm;
	delete svr;
	delete wvr;
}


int CascadeERT::init_for_image(FdImage* myimg)
{
	wvr->init_for_image(myimg);
	wvm->init_for_image(myimg);	// Theoretically, only the first would need to create the pyramid. 
	svm->init_for_image(myimg);	// Because in this case, all other detectors work on a patchlist.
	svr->init_for_image(myimg);
	return 1;
}

int CascadeERT::detect_on_image(FdImage* myimg)
{
	
	wvr->extract(myimg);
	std::vector<FdPatch*> candidates_wvr;
	std::cout << "[CascadeERT] Running WVR on whole image" << std::endl;
	candidates_wvr = wvr->detect_on_image(myimg);

	Logger->drawBoxesWithYawAngleColor(myimg->data_matbgr, candidates_wvr);
	//Logger->LogRegressorDetect(&this, &myimg, &candidates_wvr, "Hi"); // Not yet implemented
	cv::imwrite("blaaaaa.png", myimg->data_matbgr);

	std::vector<FdPatch*> candidates_wvr_center;

	int left=0; int right=0; int center=0;

	std::vector<FdPatch*>::iterator itr;
	for (itr = candidates_wvr.begin(); itr != candidates_wvr.end(); ++itr ) {
		if((*itr)->fout[wvr->getIdentifier()] < -35) {
			//l/r
			left++;
		} else if((*itr)->fout[wvr->getIdentifier()] > 35) {
			//l/r
			right++;
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

	std::sort(candidates_svm_center.begin(), candidates_svm_center.end(), FdPatch::SortByDetectorSVMCertainty());


	//Logger->drawBoxes(myimg->data_matbgr, candidates_svm_center);
	//Logger->drawYawAngle(myimg->data_matbgr, candidates_svm_center);

	// Debug output: Output all patches that passed the SVM, write their SVR value in the filename!
	std::vector<FdPatch*>::iterator itr2;
	int i=0;
	char* bla = new char[500];
	for (itr2 = candidates.begin(); itr2 != candidates.end(); ++itr2 ) {
		if((*itr2)->certainty[svm->getIdentifier()]>=0.90) {
			sprintf(bla, "face_yawout%1.3f.png", (*itr2)->fout[svr->getIdentifier()]);
			(*itr2)->writePNG(bla);
			i++;
		}
	}
	delete[] bla;
	
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
// patchDetectApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int _tmain(int argc, _TCHAR* argv[])
{
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(17534);

	char* fn_detFrontal = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat";
	char* fn_regrSVR = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_ra.mat";
	char* fn_regrWVR = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_la.mat";
	
	FdImage *myimg = new FdImage();
	//myimg->load("D:\\CloudStation\\libFD_patrik2011\\data\\firstrun\\ws_220.tiff");
	//myimg->load("D:\\CloudStation\\libFD_patrik2011\\horse.jpg");
	//myimg->load("D:\\CloudStation\\libFD_patrik2011\\ws_220.jpg");
	//myimg->load("D:\\CloudStation\\libFD_patrik2011\\ws_115.jpg");
	
	myimg->load("D:\\CloudStation\\libFD_patrik2011\\ws_220.tif");
	//myimg->load("D:\\CloudStation\\libFD_patrik2011\\ws_71.tif");
	//myimg->load("D:\\CloudStation\\libFD_patrik2011\\color_feret_00194_940128_hr.png");


	CascadeERT *ert = new CascadeERT();
	ert->wvm->load(fn_detFrontal);
	//ert->svm->load(fn_detFrontal);
	//ert->wvr->load(fn_regrWVR);
	//ert->svr->load(fn_regrSVR);

	
	std::vector<FdImage> imgs(4864);
	std::vector<FdPatch> fps(4864);
	char* fn = new char[500];

	std::vector<FdPatch*> ml_patches;
	for(int i=1; i<=15; i++) {
		sprintf(fn, "D:\\CloudStation\\libFD_patrik2011\\cp\\cp_%04d.png", i);
		imgs[i-1].load(fn);
		fps[i-1].w = imgs[i-1].w;
		fps[i-1].h = imgs[i-1].h;
		fps[i-1].data = imgs[i-1].data;	// Has a bug because this pointer can't be deleted twice
		ml_patches.push_back(&fps[i-1]);


	}

	ert->wvm->detect_on_patchvec(ml_patches);
	std::vector<FdPatch*>::iterator pit = ml_patches.begin();
	for(int i=1; i<=15; i++) {
		//Logger->drawPatchYawAngleColor(imgs[i-1].data_matbgr, *ml_patches[i-1]);
		//sprintf(fn, "D:\\CloudStation\\libFD_patrik2011\\cp_out\\cp_%04d_w.png", i);
		//cv::imwrite(fn, imgs[i-1].data_matbgr);
		std::cout << ml_patches[i-1]->fout["RegressorSVR"] << std::endl;
	}
	delete fn;
	//candidates_wvm_center = wvm->detect_on_patchvec(candidates_wvr_center);

	//ert->init_for_image(myimg);
	//ert->detect_on_image(myimg);

	delete ert;
	delete myimg;
	return 0;
}


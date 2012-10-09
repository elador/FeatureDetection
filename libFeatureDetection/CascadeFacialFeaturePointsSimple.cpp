#include "stdafx.h"
#include "CascadeFacialFeaturePointsSimple.h"


CascadeFacialFeaturePointsSimple::CascadeFacialFeaturePointsSimple(void)
{
	//leye = new CascadeWvmOeSvmOe("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_lx.mat");
	reye = new CascadeWvmOeSvmOe("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_rx.mat");
	reye->setIdentifier("reye");
	/*nosetip = new CascadeWvmOeSvmOe("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_nt.mat");
	lmouth = new CascadeWvmOeSvmOe("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_lm.mat");
	rmouth = new CascadeWvmOeSvmOe("D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_rm.mat");
	*/
}


CascadeFacialFeaturePointsSimple::~CascadeFacialFeaturePointsSimple(void)
{
	//delete leye;
	delete reye;
	/*delete nosetip;
	delete lmouth;
	delete rmouth;*/
}

int CascadeFacialFeaturePointsSimple::initForImage(FdImage* myimg)
{
	reye->initForImage(myimg);
	return 1;
}

int CascadeFacialFeaturePointsSimple::detectOnImage(FdImage* myimg)
{
	reye->detectOnImage(myimg);
	return 1;
}
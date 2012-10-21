#include "stdafx.h"
#include "CascadeFacialFeaturePointsSimple.h"


CascadeFacialFeaturePointsSimple::CascadeFacialFeaturePointsSimple(void)
{
	reye = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_re.mat");
	reye->setIdentifier("reye");
	leye = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_le.mat");
	leye->setIdentifier("leye");
	nosetip = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_nt.mat");
	nosetip->setIdentifier("nosetip");
	rmouth = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_rm.mat");
	rmouth->setIdentifier("rmouth");
	lmouth = new CascadeWvmOeSvmOe("C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_lm.mat");
	lmouth->setIdentifier("lmouth");
}


CascadeFacialFeaturePointsSimple::~CascadeFacialFeaturePointsSimple(void)
{
	delete reye;
	delete leye;
	delete nosetip;
	delete rmouth;
	delete lmouth;
}

int CascadeFacialFeaturePointsSimple::initForImage(FdImage* myimg)
{
	reye->initForImage(myimg);
	leye->initForImage(myimg);
	nosetip->initForImage(myimg);
	rmouth->initForImage(myimg);
	lmouth->initForImage(myimg);
	return 1;
}

int CascadeFacialFeaturePointsSimple::detectOnImage(FdImage* myimg)
{
	reye->detectOnImage(myimg);
	leye->detectOnImage(myimg);
	nosetip->detectOnImage(myimg);
	rmouth->detectOnImage(myimg);
	lmouth->detectOnImage(myimg);
	return 1;
}

void CascadeFacialFeaturePointsSimple::setRoiInImage(Rect roi)
{
	reye->setRoiInImage(roi);
	leye->setRoiInImage(roi);
	nosetip->setRoiInImage(roi);
	rmouth->setRoiInImage(roi);
	lmouth->setRoiInImage(roi);
}
